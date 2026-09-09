"""`milabench cherrybin prepare` — checkout selected benches from a shared .db."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from argklass.arguments import argument, group
from argklass.command import Command
from cantilever.core.timer import show_timings, timeit

from milabench.common import CommonArguments, get_multipack

from .util import (
    blob_cache_dir,
    is_full_archive_checkout,
    materialize_checkout,
    materialize_checkout_all,
    require_cherrybin,
    uses_generated_dataset,
)


def _show_timings() -> None:
    # cantilever's timer_builder is process-global and keyed by native thread
    # id; a stale, already-unwound thread entry left over from unrelated code
    # can make show_timings() crash on an empty list. The breakdown is a
    # diagnostic nice-to-have -- never let it take the actual command down.
    try:
        show_timings(force=True)
    except Exception as exc:
        print(f"[cherrybin] could not print timing breakdown: {exc}")


def _generate_argv(args, names: list) -> list:
    """Build the `milabench prepare --select ...` argv for the generated benches.

    Run out-of-process rather than on a background thread: benchmate.warden's
    pipe_warden() snapshots this *whole process's* /proc/<pid>/fd/ before and
    after running a bench's prepare script, then closes anything "new" as an
    assumed leaked pipe -- os.getpid() is shared by every thread, so it can't
    tell a real leak from a file some other thread (e.g. cherrybin's own
    reader thread, mid-checkout) opened for something unrelated. It has
    closed cherrybin's archive file out from under it this way. A real
    subprocess gets its own, isolated fd table, so pipe_warden there can
    never reach cherrybin's fds.

    `milabench prepare` already runs multiple --select'ed benches one after
    another in a single process (see MultiPackage.do_phase), so one
    subprocess for every generated bench together keeps them sequential
    relative to each other while running as a whole in parallel with the
    archive checkout.
    """
    argv = [sys.executable, "-m", "milabench", "prepare", "--select", ",".join(names)]
    if config := getattr(args, "config", None):
        argv += ["--config", config]
    if system := getattr(args, "system", None):
        argv += ["--system", system]
    if base := getattr(args, "base", None):
        argv += ["--base", base]
    if exclude := getattr(args, "exclude", None):
        argv += ["--exclude", exclude]
    for override in getattr(args, "override", None) or []:
        argv += ["--override", override]
    if capabilities := getattr(args, "capabilities", None):
        argv += ["--capabilities", capabilities]
    if getattr(args, "resume", False):
        argv += ["--resume"]
    return argv


def _print_checkout_result(result, standard_data, standard_cache) -> None:
    print(
        f"[{result.benchmark}] {result.file_count} files "
        f"({result.pulled_from_archive} from archive, "
        f"{result.already_cached} cached, "
        f"{result.chunks_read} chunks, "
        f"{result.archive_bytes_read / 1e6:.1f} MB read) "
        f"-> {standard_data} / {standard_cache}, "
        f"{result.io.summary()}"
    )


class Prepare(Command):
    """Checkout datasets/checkpoints for selected benchmarks from a cherrybin archive.

    Drop-in replacement for `milabench prepare` on compute nodes without
    internet: files are read from --shared (a sqlite file) and hardlinked
    into the standard data/ and cache/ directories.
    """

    name = "prepare"

    # fmt: off
    @dataclass
    class Arguments:
        """Checkout datasets from a cherrybin archive."""
        common : CommonArguments = group(CommonArguments)
        shared : str             = argument("--shared", default="")  # Path to the cherrybin archive .db
        cache  : str             = ""                    # Blob cache directory
        io_chunk : int           = 16 * 1024 * 1024      # Stream I/O chunk size in bytes
        no_stream: bool          = False                 # Use per-file naive checkout
    # fmt: on

    @staticmethod
    def execute(args):
        require_cherrybin()
        if not args.shared:
            print("error: --shared /path/to/archive.db is required")
            return 1
        if not os.path.exists(args.shared):
            print(f"error: archive not found: {args.shared}")
            return 1

        mp = get_multipack(args, run_name="cherrybin.prepare.{time}")
        blob_cache = args.cache or blob_cache_dir(args.base)
        stream = not args.no_stream
        io_chunk = getattr(args, "io_chunk", None)

        archive_packs = {
            name: pack
            for name, pack in mp.packs.items()
            if not uses_generated_dataset(pack)
        }
        generated_packs = {
            name: pack
            for name, pack in mp.packs.items()
            if uses_generated_dataset(pack)
        }
        errors = 0

        generate_proc = None
        generate_timer = None
        if generated_packs:
            argv = _generate_argv(args, sorted(generated_packs))
            print(f"[cherrybin] preparing generated datasets in a subprocess: {' '.join(argv)}")
            generate_timer = timeit("cherrybin.generate_local")
            generate_timer.__enter__()
            generate_proc = subprocess.Popen(argv)

        try:
            if archive_packs and is_full_archive_checkout(
                args.shared, sorted(archive_packs)
            ):
                mode = "naive" if args.no_stream else "stream"
                print(
                    f"[cherrybin] full archive checkout "
                    f"({len(archive_packs)} benchmarks, {mode})"
                )
                try:
                    with tempfile.TemporaryDirectory() as staging:
                        results = materialize_checkout_all(
                            args.shared,
                            archive_packs,
                            blob_cache,
                            staging,
                            io_chunk=io_chunk,
                            stream=stream,
                        )
                except (FileNotFoundError, KeyError) as exc:
                    print(f"error: {exc}")
                    errors += 1
                else:
                    for name, pack in archive_packs.items():
                        result = results[name]
                        _print_checkout_result(
                            result, pack.dirs.data, pack.dirs.cache
                        )
            else:
                for name, pack in archive_packs.items():
                    standard_data = pack.dirs.data
                    standard_cache = pack.dirs.cache
                    isolated_data = standard_data / name
                    isolated_cache = standard_cache / name
                    try:
                        with tempfile.TemporaryDirectory() as staging:
                            result = materialize_checkout(
                                args.shared,
                                name,
                                standard_data,
                                standard_cache,
                                isolated_data,
                                isolated_cache,
                                blob_cache,
                                staging,
                                io_chunk=io_chunk,
                                stream=stream,
                            )
                    except (FileNotFoundError, KeyError) as exc:
                        print(f"error: {exc}")
                        errors += 1
                        continue
                    _print_checkout_result(result, standard_data, standard_cache)
        finally:
            # generate_rc is only trustworthy once the subprocess has
            # actually exited -- must wait before it factors into the exit
            # code below.
            generate_rc = 0
            if generate_proc is not None:
                generate_rc = generate_proc.wait()
                if generate_rc:
                    print(f"[cherrybin] generated dataset prepare failed (exit {generate_rc})")
            if generate_timer is not None:
                generate_timer.__exit__(None, None, None)
            _show_timings()

        return 1 if (errors or generate_rc) else 0


COMMANDS = Prepare
