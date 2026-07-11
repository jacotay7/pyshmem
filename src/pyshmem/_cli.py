"""Command-line interface for pyshmem stream management."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyshmem",
        description="Manage pyshmem shared-memory streams.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    unlink_p = sub.add_parser(
        "unlink",
        help="Unlink (destroy) one or more named streams by user-given name.",
    )
    unlink_p.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help=(
            "User-visible stream names to unlink (the name passed to "
            "pyshmem.create, as shown by 'list')."
        ),
    )

    sub.add_parser(
        "list",
        help="List the user-visible names of all existing pyshmem streams.",
    )

    purge_p = sub.add_parser(
        "purge",
        help="Remove ALL pyshmem segments from shared memory.",
    )
    purge_p.add_argument(
        "--include-cuda-orphans",
        action="store_true",
        help=(
            "also remove dead-producer cuda.shm.* files from any PyTorch "
            "application owned by this OS account"
        ),
    )

    args = parser.parse_args(argv)

    if args.command == "unlink":
        import pyshmem

        for name in args.names:
            pyshmem.unlink(name)
            print(f"unlinked {name!r}")
        return 0

    if args.command == "list":
        import pyshmem

        streams = pyshmem.list_streams()
        if streams:
            for name in streams:
                print(name)
        else:
            print("no streams found", file=sys.stderr)
        return 0

    if args.command == "purge":
        import pyshmem

        removed = pyshmem.purge(include_cuda_orphans=args.include_cuda_orphans)
        if removed:
            for name in removed:
                print(f"purged {name!r}")
            print(f"removed {len(removed)} stream(s)")
        else:
            print("no streams found", file=sys.stderr)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
