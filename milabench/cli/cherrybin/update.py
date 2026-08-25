"""`milabench cherrybin update` — prepare into per-pack trees and write the .db."""

from __future__ import annotations

from dataclasses import dataclass

from argklass.arguments import argument, group
from argklass.command import Command

from milabench.common import CommonArguments, get_multipack

from .util import (
    isolate_pack_dirs,
    mirror_isolated_to_standard,
    pack_roots,
    prepare_locally,
    remove_isolated_trees,
    require_cherrybin,
    uses_generated_dataset,
)


class Update(Command):
    """Create or update a cherrybin archive after preparing selected benchmarks.

    For each selected pack, in turn: prepare into isolated data/<name>
    and cache/<name>, write that bench into --shared, then move on.
    One massive bench is indexed before the next download starts.
    Hardlinks into the standard data/cache dirs so `milabench run` works,
    unless --clean, which deletes the isolated download after it is in the db.
    """

    name = "update"

    # fmt: off
    @dataclass
    class Arguments:
        """Create or update a cherrybin archive."""
        common    : CommonArguments = group(CommonArguments)
        shared    : str             = argument("--shared", default="")  # Path to the cherrybin archive .db
        clean     : bool            = False                 # Delete isolated downloads after they are in the db
        io_chunk  : int             = 4 * 1024 * 1024       # Stream I/O chunk size in bytes
        shortrace : bool            = False                 # On error show short stacktrace
    # fmt: on

    @staticmethod
    def execute(args):
        core = require_cherrybin()
        if not args.shared:
            print("error: --shared /path/to/archive.db is required")
            return 1

        mp = get_multipack(args, run_name="cherrybin.update.{time}")

        # One bench at a time so a huge prepare is flushed to the archive
        # before the next download starts.
        for name, pack in mp.packs.items():
            if uses_generated_dataset(pack):
                print(f"[{pack.config['name']}] generated dataset, skip archive")
                continue

            orig_data, orig_cache = isolate_pack_dirs(pack)
            ret = prepare_locally(pack, shortrace=args.shortrace)
            if ret:
                return ret

            stats = core.update_file(
                args.shared,
                pack.config["name"],
                pack_roots(pack),
                io_chunk=getattr(args, "io_chunk", None),
            )
            print(
                f"[{stats.name}] {stats.file_count} files "
                f"(+{stats.added} -{stats.removed} ={stats.unchanged}) "
                f"{'updated' if stats.changed else 'unchanged'} {args.shared}"
            )
            if args.clean:
                # Skip mirroring: hardlinks would keep the bytes on disk.
                remove_isolated_trees(pack)
            else:
                mirror_isolated_to_standard(pack, orig_data, orig_cache)

        return 0


COMMANDS = Update
