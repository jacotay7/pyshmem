Platform Notes
==============

Linux and macOS
---------------

POSIX platforms support the lifetime semantics expected by the CPU test suite,
including reopening a segment after the original creator exits as long as the
segment still exists.

Windows
-------

Windows inherits a hard limitation from ``multiprocessing.shared_memory``:
the operating system destroys the shared-memory block when the last handle is
closed.

Consequences:

- a segment cannot outlive its creator if no other handle remains open
- ``close()`` followed by ``pyshmem.open(...)`` fails if that close call dropped
  the final live handle

This is an operating-system behavior, not a pyshmem-specific policy.