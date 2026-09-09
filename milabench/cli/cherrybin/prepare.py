"""`milabench cherrybin prepare` — checkout selected benches from a shared .db."""

from __future__ import annotations

import os
import tempfile
import threading
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
    prepare_locally,
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


def _prepare_generated(generated_packs: dict, failed: list) -> None:
    """Run the (sequential) local prepare for generated-dataset benches.

    Meant to run on a background thread alongside the archive checkout.
    Left sequential on purpose: benchmate.warden's children_warden tracks
    subprocesses by os.getpid(), which is shared by every thread in this
    process -- two of these running concurrently in separate threads could
    see each other's still-running subprocess as an "unexpected leftover
    child" and kill it. The checkout below never touches warden/signal
    machinery at all, so pairing one sequential prepare thread with the
    checkout on the main thread is safe; running multiple prepares
    concurrently would not be.
    """
    with timeit("cherrybin.generate_local"):
        for name, pack in generated_packs.items():
            print(f"[{name}] generated dataset, running prepare")
            if prepare_locally(pack, shortrace=False):
                failed.append(name)


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
        generate_failed: list = []

        generate_thread = None
        if generated_packs:
            generate_thread = threading.Thread(
                target=_prepare_generated,
                args=(generated_packs, generate_failed),
                daemon=True,
            )
            generate_thread.start()

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
            # generate_failed is only trustworthy once the background thread
            # has actually finished -- must join before it factors into the
            # exit code below.
            if generate_thread is not None:
                generate_thread.join()
            for name in generate_failed:
                print(f"[{name}] generated dataset prepare failed")
            _show_timings()

        return 1 if (errors or generate_failed) else 0


COMMANDS = Prepare
