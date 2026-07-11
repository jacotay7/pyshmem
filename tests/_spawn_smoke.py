"""Standalone cross-process smoke check (not collected by pytest).

Runs a create -> spawn -> open/read round trip outside the pytest harness so a
platform-specific hang (notably Windows spawn) can be attributed to pyshmem
itself rather than the test runner, and so a hung child dumps its own traceback
instead of silently wedging CI.

Run directly::

    python tests/_spawn_smoke.py
"""

from __future__ import annotations

import faulthandler
import multiprocessing as mp
import sys

import numpy as np


def _consumer(name: str, queue) -> None:
    # If the child wedges (e.g. during import or open), dump its traceback and
    # exit non-zero after 20s rather than blocking the parent forever.
    faulthandler.dump_traceback_later(20, exit=True)
    import pyshmem

    shm = pyshmem.open(name)
    try:
        queue.put(shm.read().tolist())
    finally:
        shm.close()


def main() -> int:
    faulthandler.enable()
    import pyshmem

    name = "pyshmem_spawn_smoke"
    ctx = mp.get_context("spawn")
    shm = pyshmem.create(name, shape=(4,), dtype=np.float32)
    try:
        shm.write(np.arange(4, dtype=np.float32))
        queue = ctx.Queue()
        process = ctx.Process(target=_consumer, args=(name, queue))
        process.start()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            print("SMOKE FAIL: child still alive after 30s", file=sys.stderr)
            return 1
        if process.exitcode != 0:
            print(
                f"SMOKE FAIL: child exitcode {process.exitcode}",
                file=sys.stderr,
            )
            return 1
        values = queue.get(timeout=5)
        if values != [0.0, 1.0, 2.0, 3.0]:
            print(f"SMOKE FAIL: unexpected payload {values}", file=sys.stderr)
            return 1
        print(f"SMOKE OK: {values}")
        return 0
    finally:
        shm.unlink()


if __name__ == "__main__":
    sys.exit(main())
