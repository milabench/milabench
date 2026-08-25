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
    materialize_checkout,
    prepare_locally,
    require_cherrybin,
    uses_generated_dataset,
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

        errors = 0
        for pack in mp.packs.values():
            name = pack.config["name"]
            if uses_generated_dataset(pack):
                print(f"[{name}] generated dataset, running prepare")
                ret = prepare_locally(pack, shortrace=False)
                if ret:
                    errors += 1
                continue

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
                        io_chunk=getattr(args, "io_chunk", None),
                    )
            except (FileNotFoundError, KeyError) as exc:
                print(f"error: {exc}")
                errors += 1
                continue
            print(
                f"[{result.benchmark}] {result.file_count} files "
                f"({result.pulled_from_archive} from archive, "
                f"{result.already_cached} cached) -> {standard_data} / {standard_cache}"
            )

        return 1 if errors else 0


COMMANDS = Prepare
