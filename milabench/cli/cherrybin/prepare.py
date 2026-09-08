"""`milabench cherrybin prepare` — checkout selected benches from a shared .db."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from argklass.arguments import argument, group
from argklass.command import Command

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
        io_chunk : int           = 4 * 1024 * 1024       # Stream I/O chunk size in bytes
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
        errors = 0

        for name, pack in mp.packs.items():
            if uses_generated_dataset(pack):
                print(f"[{name}] generated dataset, running prepare")
                ret = prepare_locally(pack, shortrace=False)
                if ret:
                    errors += 1

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
                return 1

            for name, pack in archive_packs.items():
                result = results[name]
                _print_checkout_result(
                    result, pack.dirs.data, pack.dirs.cache
                )
            return 1 if errors else 0

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

        return 1 if errors else 0


COMMANDS = Prepare
