Shared-Memory Format
====================

pyshmem streams use separate named segments for payload data, metadata, and an
optional PyTorch CUDA IPC reduction. The metadata segment is the stable entry
point used by :func:`pyshmem.open` to reconstruct shape, dtype, backend, and
lifecycle state.

Version 3 metadata header
-------------------------

New streams use a 256-byte, explicitly little-endian header followed by a
256-byte UTF-8 name region. Integer fields have fixed widths and the frequently
updated counters are aligned to eight-byte boundaries.

.. list-table::
   :header-rows: 1

   * - Field
     - Offset
     - Type
     - Meaning
   * - magic
     - 0
     - 8 bytes
     - ``PYSHMEM\\0``
   * - version
     - 8
     - uint16
     - Format version, currently 3
   * - header_size
     - 10
     - uint16
     - Header size, currently 256
   * - flags
     - 12
     - uint32
     - GPU, CPU-mirror, instance-id, header-CRC, and notify feature bits
   * - dtype_code
     - 16
     - uint16
     - Index in pyshmem's dtype table
   * - ndim
     - 18
     - uint16
     - Number of shape dimensions
   * - device_index
     - 20
     - int32
     - CUDA device, or -1 for CPU
   * - creator_pid
     - 24
     - int64
     - Process that created the stream
   * - size
     - 32
     - uint64
     - Payload size in bytes
   * - count
     - 40
     - uint64
     - Completed-write count
   * - write_sequence
     - 48
     - int64
     - Even stable, odd writing, negative invalid
   * - write_time
     - 56
     - float64
     - UNIX time of the last completed write
   * - lock_owner_pid
     - 64
     - int64
     - Diagnostic/current writer process
   * - lock_depth
     - 72
     - uint32
     - Diagnostic re-entrant lock depth
   * - instance_id
     - 76
     - 16 bytes
     - Random identity for this name generation
   * - header_crc
     - 92
     - uint32
     - CRC-32 of the immutable header fields plus the name region
   * - reserved
     - 96
     - 8 bytes
     - Must be zero; reserved for extensions
   * - shape
     - 104
     - uint64[19]
     - Positive dimensions, unused entries zero

The name region begins at byte 256 and contains at most 256 UTF-8 bytes,
null-padded. Discovery and ordinary purge accept a segment only when this name
hashes back to the exact internal segment identifier.

Compatibility
-------------

The stable dtype code table contains signed and unsigned 8/16/32/64-bit
integers, float16/32/64, bool, and complex64/128. Bool and complex were appended
to the original table, preserving every existing code. GPU acceptance is a
runtime PyTorch capability decision; the on-disk code itself remains stable.

Version 3 readers retain read/write compatibility with version 2 metadata,
whose first 256 bytes are a native ``float64[32]`` array. New streams are always
created as version 3. Unknown versions, invalid v3 header sizes, and malformed
headers are rejected rather than guessed. Version 2 support is intended for
attaching to streams left by pyshmem 1.0.x; there is no in-place conversion.

Before mapping a payload, ``open()`` validates the metadata segment length,
known flags, zeroed reserved bytes, UTF-8 name/hash relationship, dtype code,
dimension bounds, positive shape, zeroed unused dimensions, exact
shape/dtype/size product, CPU/GPU device rules, creator PID, timestamps, lock
state, and that the data segment is large enough for the declared payload.
Segment-length checks require a sufficient (not exact) mapping because macOS
rounds shared-memory allocations up to a page. Discovery and purge apply the
same header validation and ignore candidates that fail it.

As a final integrity backstop, streams carrying the header-CRC feature flag
also validate ``header_crc``: a CRC-32 computed once at creation over the
immutable header fields (everything except the mutating ``count``,
``write_sequence``, ``write_time``, ``lock_owner_pid``, and ``lock_depth``
counters, and the CRC slot itself) together with the 256-byte name region.
Because the hot-path counters are excluded, ordinary writes and lock activity
never disturb it. The granular field checks above give precise diagnostics for
individual malformed fields; the CRC additionally rejects silent bit-flips or
torn header writes that leave every field individually plausible. Version 2
streams and early version 3 streams predating the flag carry no CRC and skip
this check.

The random ``instance_id`` distinguishes successive streams created under the
same user-visible name. Handle-level unlink verifies this identity while
holding the persistent per-name lifecycle lock, so an old handle cannot destroy
a newer replacement. Version 2 streams and early version 3 streams without the
feature flag remain readable, but cannot receive this generation protection.

Memory ordering
---------------

This section specifies the interprocess memory model pyshmem relies on, the
guarantees it does and does not provide, and the platforms on which the model
has been validated.

Encoding and alignment
~~~~~~~~~~~~~~~~~~~~~~~~

Every integer field is little-endian with a fixed width (see the header table
above). The two fields mutated on the hot path — ``count`` (offset 40) and
``write_sequence`` (offset 48) — are 8-byte-wide integers placed on 8-byte
boundaries. Shared-memory segments begin on a page boundary, so these offsets
are absolutely aligned in every mapping, not merely aligned within the header.
Natural alignment is what makes a single 64-bit store or load indivisible on
the supported ISAs, and pyshmem enforces the alignment of these offsets as a
format invariant.

Publication protocol (seqlock)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``write_sequence`` is a sequence lock:

- **even** — the payload is stable and readable;
- **odd** — a write is in progress; readers must wait;
- **negative** — the last write failed or its writer died; the payload is
  invalid and reads raise :class:`~pyshmem.InconsistentStreamError` until a
  later complete write starts a fresh generation.

A writer increments the sequence to odd, copies the payload, then increments it
to even. A lock-free reader waits for an even sequence, snapshots the payload,
then re-reads the sequence; an unchanged even value means the snapshot did not
overlap a write. This is the standard seqlock retry loop
(:meth:`_read_consistent_cpu` / :meth:`_read_consistent_gpu`).

What pyshmem relies on
~~~~~~~~~~~~~~~~~~~~~~~~

- **Single-writer serialization.** Concurrent writers are excluded by the
  cross-process ``portalocker`` file lock, so only one process ever advances the
  sequence at a time.
- **Architecture-specific publication barriers.** On x86-64, naturally aligned
  stores plus TSO ordering provide the required publication order without FFI
  overhead. Elsewhere pyshmem uses GCC ``libatomic`` acquire/release operations
  when available. The odd publication is sequentially consistent and the final
  even publication is a release store sampled with acquire semantics.
- **Portable fallback.** If neither x86-64 TSO nor ``libatomic`` is available,
  readers briefly acquire and release the process-shared OS file lock before
  both sequence samples. Writers hold that lock across payload publication, so
  the syscalls supply the synchronization edge while the payload copy itself
  remains outside the reader lock.
- **Overlap detection.** The post-snapshot sequence sample detects a write that
  raced with the copy, so the snapshot is retried.

What pyshmem does **not** provide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The portable fallback is not lock-free at the two sequence-sampling points, and
``libatomic`` is discovered at runtime rather than required as a binary
dependency. The payload copy remains outside the lock and keeps seqlock retry
semantics. Public ``count`` and ``write_sequence`` properties are diagnostic
snapshots; safe reads use the synchronized publication path.

Validated platforms
~~~~~~~~~~~~~~~~~~~~~

The direct TSO path is used only on **x86-64**. The native acquire/release path
is used wherever ``libatomic`` is available, including validated Linux
**aarch64**. Other POSIX architectures use the OS-lock fallback rather than
silently weakening consistency.

Payload and CUDA handle segments
--------------------------------

The CPU payload is a contiguous C-order array whose byte count must equal the
product of shape and dtype item size. GPU streams additionally store a serialized
PyTorch reduction in the ``_gpu`` segment; this is an implementation-dependent,
trusted-process interface rather than part of the stable metadata format.

The ``_gpu`` segment holds torch's official ``reduce_tensor`` output
(``rebuild_cuda_tensor`` plus primitive arguments) pickled for cross-process
tensor reconstruction. Because the segment is writable (mode 0600, so exposure
is limited to the same OS account), pyshmem deserializes it with a **restricted
unpickler** that only resolves torch's known CUDA rebuild globals and inert
dtype values. A tampered payload raises ``UnpicklingError`` instead of executing
arbitrary code, so the trust boundary is the set of processes that can write the
segment (same-account producers), not any code they could smuggle into it.
Reconstruction still assumes a live, trusted producer for the CUDA IPC handle
itself.
