"""Run several configurations inside a single benchmark process.

Some benchmarks take a long time to start (imports, model download, compilation,
...). Instead of paying that cost once per configuration, the benchmark can
switch configuration while it is running.

Configurations come from the benchmark config (``MILABENCH_CONFIG`` env var),
under the key ``multiconfig``: a list of argv, each one is a configuration to
run in the same process.

.. code-block:: yaml

    mybench:
      multiconfig:
        - ["--batch-size", "32"]
        - ["--batch-size", "64"]

Usage in a benchmark:

.. code-block:: python

    from benchmate.config import multiconfig

    for argv in multiconfig():
        with argv:                    # a crash here moves on to the next config
            args = parser.parse_args(argv)
            train(args)               # may call next_config() to jump out early

    # or equivalently
    run_multiconfig(lambda argv: train(parser.parse_args(argv)))

Every metric pushed through ``benchmate.metrics`` carries a ``config_key``
(hash of the config) so metrics can be mapped back to the config that
produced them.
"""

import hashlib
import json
import os
import sys
import traceback


class ConfigChange(BaseException):
    """Abort the current configuration and move on to the next one.

    It derives from ``BaseException`` so it is an unconditional jump that
    ``except Exception`` handlers do not swallow. It is not a ``SystemExit``
    so code handling ``sys.exit()`` (argparse wrappers, voir, ...) does not
    swallow it either, and a real exit stays distinguishable from a config change.
    """

    def __init__(self, reason=None):
        super().__init__(reason)
        self.reason = reason


# Plain global on purpose, not a ContextVar: a ContextVar is not inherited by
# threads started before the value is set (and threads start with an empty
# context), so metrics pushed from monitor/sampler threads would lose their
# config_key. Only one config runs at a time, so a process wide value is correct.
_CONFIG_KEY = None


def get_config_key():
    """Key of the config currently running, ``None`` outside of multiconfig"""
    return _CONFIG_KEY


def compute_config_key(config):
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def set_config_key(config):
    """Make ``config`` the current config, returns its key.

    ``None`` clears the key. Start and end markers are printed to stderr.
    """
    global _CONFIG_KEY

    if _CONFIG_KEY is not None:
        print(f"[multiconfig] config end ({_CONFIG_KEY})", file=sys.stderr, flush=True)

    _CONFIG_KEY = None if config is None else compute_config_key(config)

    if _CONFIG_KEY is not None:
        print(
            f"[multiconfig] config start ({_CONFIG_KEY}): {config}",
            file=sys.stderr,
            flush=True,
        )
    return _CONFIG_KEY


def tag_metric(data):
    """Add the current config key to a metric (in place)"""
    key = _CONFIG_KEY
    if key is not None:
        data.setdefault("config_key", key)
    return data


def next_config(reason=None):
    """Stop the current config, the next one (if any) will start.

    Does not return.
    """
    print(
        f"[multiconfig] config change triggered ({get_config_key()}): {reason}",
        file=sys.stderr,
        flush=True,
    )
    raise ConfigChange(reason)


def load_multiconfig(environ=None):
    """Read the list of argv from ``MILABENCH_CONFIG["multiconfig"]``"""
    environ = os.environ if environ is None else environ
    raw = environ.get("MILABENCH_CONFIG")
    if not raw:
        return []

    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        return []

    configs = cfg.get("multiconfig") if isinstance(cfg, dict) else None
    return list(configs or [])


class Config(list):
    """One configuration (an argv).

    Used as a context manager it runs the body for this config:
    the config key is set, a ``ConfigChange`` or any crash is reported
    and swallowed so the caller can move on to the next config.
    ``SystemExit``/``KeyboardInterrupt`` and voir's stop signal still propagate.
    """

    @property
    def key(self):
        return compute_config_key(list(self))

    def __enter__(self):
        set_config_key(list(self))
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                return False

            if issubclass(exc_type, ConfigChange):
                return True

            if issubclass(exc_type, Exception):
                print(f"[multiconfig] {self.key} crashed, moving on", file=sys.stderr)
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
                return True

            return False
        finally:
            set_config_key(None)


def multiconfig(configs=None):
    """Iterate over the configs to run, defaults to ``multiconfig`` of the bench config.

    Without any ``multiconfig`` a single config is yielded: the process argv.
    """
    if configs is None:
        configs = load_multiconfig()

    if not configs:
        configs = [sys.argv[1:]]

    for argv in configs:
        if isinstance(argv, str):
            argv = argv.split()
        yield Config(argv)


def run_multiconfig(fn, configs=None):
    """Call ``fn(argv)`` for each config, a failing config does not stop the others"""
    for argv in multiconfig(configs):
        with argv:
            fn(list(argv))
