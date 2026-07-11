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
     - GPU and CPU-mirror feature bits
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
   * - reserved
     - 76
     - 28 bytes
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

Version 3 readers retain read/write compatibility with version 2 metadata,
whose first 256 bytes are a native ``float64[32]`` array. New streams are always
created as version 3. Unknown versions, invalid v3 header sizes, and malformed
headers are rejected rather than guessed. Version 2 support is intended for
attaching to streams left by pyshmem 1.0.x; there is no in-place conversion.

Before mapping a payload, ``open()`` validates the metadata segment length,
known flags, zeroed reserved bytes, UTF-8 name/hash relationship, dtype code,
dimension bounds, positive shape, zeroed unused dimensions, exact
shape/dtype/size product, CPU/GPU device rules, creator PID, timestamps, lock
state, and the actual data-segment byte length. Discovery and purge apply the
same header validation and ignore candidates that fail it.

Memory ordering
---------------

Fixed-width representation does **not** by itself make NumPy loads and stores
interprocess atomics. Writers are serialized by the cross-process file lock and
the sequence protocol detects overlapping reads, but a fully specified
acquire/release implementation for lock-free readers remains future work. The
format deliberately aligns counters so a native atomic backend can be added
without another layout change.

Payload and CUDA handle segments
--------------------------------

The CPU payload is a contiguous C-order array whose byte count must equal the
product of shape and dtype item size. GPU streams additionally store a serialized
PyTorch reduction in the ``_gpu`` segment; this is an implementation-dependent,
trusted-process interface rather than part of the stable metadata format.
