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

More configs can be added while the benchmark is running by appending to
``runtime_config.yaml`` (see ``add_runtime_config``). The file is reloaded each
time a config finishes and entries that have not run yet are queued.

Every metric pushed through ``benchmate.metrics`` carries a ``config_key``
(hash of the config) so metrics can be mapped back to the config that
produced them.
"""

import hashlib
import json
import os
import sys
import traceback
from collections import deque


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


RUNTIME_CONFIG_NAME = "runtime_config.yaml"
LOCK_TIMEOUT = 1


def runtime_config_path(environ=None):
    """Location of the runtime config file, ``None`` if it cannot be determined.

    ``MILABENCH_RUNTIME_CONFIG`` wins, else ``$MILABENCH_DIR_RUNS/<bench>/runtime_config.yaml``
    """
    environ = os.environ if environ is None else environ

    explicit = environ.get("MILABENCH_RUNTIME_CONFIG")
    if explicit:
        return explicit

    runs = environ.get("MILABENCH_DIR_RUNS")
    if not runs:
        return None

    name = None
    try:
        name = json.loads(environ.get("MILABENCH_CONFIG") or "{}").get("name")
    except (json.JSONDecodeError, AttributeError):
        pass

    return os.path.join(runs, name or "", RUNTIME_CONFIG_NAME)


def _lock(path, timeout):
    from filelock import FileLock

    return FileLock(path + ".lock", timeout=timeout)


def _normalize(argv):
    if isinstance(argv, str):
        argv = argv.split()
    return [str(a) for a in argv]


def load_runtime_config(path):
    """Read the argv list from the runtime config file.

    Never raises: a missing, locked or malformed file gives an empty list
    (with a warning when it is not simply missing).
    """
    if not path or not os.path.exists(path):
        return []

    try:
        import yaml

        with _lock(path, LOCK_TIMEOUT):
            with open(path) as fp:
                data = yaml.safe_load(fp)

        configs = data.get("multiconfig") if isinstance(data, dict) else None
        if configs is None:
            return []
        if not isinstance(configs, list):
            raise ValueError("multiconfig should be a list")

        return [_normalize(argv) for argv in configs]

    except Exception as err:
        print(f"[multiconfig] ignoring {path}: {err!r}", file=sys.stderr)
        return []


def add_runtime_config(argv, path=None):
    """Append a config to the runtime config file, to be picked up by a running benchmark."""
    add_runtime_configs([argv], path)


def add_runtime_configs(argvs, path=None):
    """Append configs to the runtime config file, to be picked up by a running benchmark.

    The write is done under the lock and atomically (temp file + rename)
    so a reader never sees a partial file.
    """
    import yaml

    path = path or runtime_config_path()
    if path is None:
        raise ValueError("No runtime config path, set MILABENCH_RUNTIME_CONFIG")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with _lock(path, 10):
        configs = []
        if os.path.exists(path):
            with open(path) as fp:
                data = yaml.safe_load(fp) or {}
            configs = list(data.get("multiconfig") or [])

        configs.extend(_normalize(argv) for argv in argvs)

        tmp = f"{path}.tmp"
        with open(tmp, "w") as fp:
            yaml.safe_dump({"multiconfig": configs}, fp)
        os.replace(tmp, path)


def sync_configs(configs):
    """Hook to make every rank agree on the configs to run.

    No-op by default, distributed benchmarks should override it (pass ``sync=``
    to ``multiconfig``) to broadcast rank 0's list, otherwise ranks reading the
    file at different times could run different configs.
    """
    return configs


def multiconfig(configs=None, runtime_config=None, sync=sync_configs):
    """Iterate over the configs to run, defaults to ``multiconfig`` of the bench config.

    Without any ``multiconfig`` a single config is yielded: the process argv.

    After each config the runtime config file is reloaded, entries that were
    not already queued or run (by config key) are appended to the queue.
    ``runtime_config`` is the file path, by default ``runtime_config_path()``
    when ``configs`` is not given, else no runtime config.
    """
    if configs is None:
        configs = load_multiconfig()
        if runtime_config is None:
            runtime_config = runtime_config_path()

    if not configs:
        configs = [sys.argv[1:]]

    pending = deque(Config(_normalize(argv)) for argv in configs)
    known = {c.key for c in pending}

    while True:
        if runtime_config:
            for argv in sync(load_runtime_config(runtime_config)):
                config = Config(_normalize(argv))
                if config.key not in known:
                    known.add(config.key)
                    pending.append(config)

        if not pending:
            return

        yield pending.popleft()


def run_multiconfig(fn, configs=None, runtime_config=None):
    """Call ``fn(argv)`` for each config, a failing config does not stop the others"""
    for argv in multiconfig(configs, runtime_config):
        with argv:
            fn(list(argv))
