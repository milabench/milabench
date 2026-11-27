"""`milabench cherrybin update` — prepare into per-pack trees and write the .db."""

from __future__ import annotations

from dataclasses import dataclass

from argklass.arguments import argument, group
from argklass.command import Command

from milabench.common import CommonArguments, get_multipack

from .util import (
    isolate_pack_dirs,
    load_fingerprints,
    pack_fingerprint,
    save_fingerprint,
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

    --only-stale skips the benches whose fingerprint (code folder + argv, see
    pack_fingerprint) matches the one recorded when they were last written to the
    archive (<archive>.fingerprints.json); a bench without a record is prepared.
    --mark-only records the current fingerprints without preparing anything (to
    start the records on an archive that is already up to date).
    """

    name = "update"

    # fmt: off
    @dataclass
    class Arguments:
        """Create or update a cherrybin archive."""
        common    : CommonArguments = group(CommonArguments)
        shared    : str             = argument("--shared", default="")  # Path to the cherrybin archive .db
        clean     : bool            = False                 # Delete isolated downloads after they are in the db
        io_chunk  : int             = 16 * 1024 * 1024      # Stream I/O chunk size in bytes
        shortrace : bool            = False                 # On error show short stacktrace
        only_stale: bool            = False                 # Skip the benches whose fingerprint did not change
        mark_only : bool            = False                 # Record the fingerprints, prepare nothing
    # fmt: on

    @staticmethod
    def execute(args):
        core = require_cherrybin()
        if not args.shared:
            print("error: --shared /path/to/archive.db is required")
            return 1

        mp = get_multipack(args, run_name="cherrybin.update.{time}")
        recorded = load_fingerprints(args.shared)

        # One bench at a time so a huge prepare is flushed to the archive
        # before the next download starts.
        for name, pack in mp.packs.items():
            if uses_generated_dataset(pack):
                print(f"[{pack.config['name']}] generated dataset, skip archive")
                continue

            bench = pack.config["name"]
            fingerprint = pack_fingerprint(pack)
            if args.mark_only:
                save_fingerprint(args.shared, bench, fingerprint)
                print(f"[{bench}] fingerprint recorded")
                continue
            if args.only_stale and recorded.get(bench) == fingerprint:
                print(f"[{bench}] unchanged since it was archived, skip")
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
                f"[{stats.name}] {stats.summary()} "
                f"{'updated' if stats.changed else 'unchanged'} {args.shared}, "
                f"{stats.io.summary()}"
            )
            save_fingerprint(args.shared, bench, fingerprint)
            if args.clean:
                # Skip mirroring: hardlinks would keep the bytes on disk.
                remove_isolated_trees(pack)
            else:
                mirror_isolated_to_standard(pack, orig_data, orig_cache)

        return 0


COMMANDS = Update
