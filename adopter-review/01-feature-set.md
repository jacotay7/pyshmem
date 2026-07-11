# Feature-set critique

## Implementation update

The first remediation batch corrected inconsistent GPU safe-read sequencing,
made unsupported GPU/unsafe `out=` usage explicit, documented automatic GPU
attachment consistently, and added bounded/failed-write read behavior. The
larger feature conclusions below are unchanged: the primitive is still a
fixed-shape capacity-one latest-value exchange, not a FIFO stream, and dtype,
backend, notification, namespace, and variable-size support remain future work.

## What the library really provides

The implemented primitive is a named, fixed-shape, fixed-dtype mutable slot with
metadata, one writer lock, lock-free copied reads, a write counter, and optional
CUDA backing. This is a coherent feature set for "publish the newest frame and
let readers sample it." CPU and GPU use similar object methods, and consumers
can attach from independent processes without separately communicating shape or
dtype.

Calling the primitive a stream obscures consequential semantics:

- capacity is always one;
- a fast producer overwrites data before a slow consumer sees it;
- there is no FIFO ordering, retention, replay, backpressure, or acknowledgement;
- there are no per-consumer cursors;
- `count` lets a caller calculate a gap manually, but `read_new()` does not
  return the count or number of skipped writes;
- there is no waitable notification; readers poll shared metadata;
- there is no end-of-stream, cancellation, schema evolution, or producer-health
  concept.

The repository should explicitly market a **latest-value shared tensor** or add
a ring-buffer/queue primitive. A capacity-N mode with monotonically numbered
slots, overrun reporting, and selectable drop/block policy would dramatically
expand adoption in camera, telemetry, and inference pipelines.

## Unified API: useful, but only partially unified

CPU and GPU share method names, but their contracts differ:

- CPU `read(safe=True)` uses a sequence-checked copy; default no-mirror GPU
  `read(safe=True)` performs a clone without sequence checks.
- `read(out=...)` is implemented only for CPU. GPU silently ignores `out`
  rather than rejecting it or supporting a tensor destination.
- GPU memory depends on a live producer allocation. A no-mirror CUDA stream
  cannot provide CPU-like persistence after its creator exits. A mirrored
  stream can fall back only to the host copy.
- GPU is CUDA/PyTorch-specific. There is no CuPy array, DLPack-facing API,
  ROCm, MPS, Intel XPU, or generic array-API backend.
- A stream is tied to its creator CUDA device. There is no peer-device attach,
  copy policy, or multi-GPU topology support.

The current surface is better described as one API façade over two transports,
not identical CPU/GPU semantics.

## Data-model gaps

The CPU dtype table supports only 11 integer/floating dtypes. Missing common
cases include `bool`, complex values, byte/string payloads, structured dtypes,
datetime/timedelta, and scalar shape `()`. The user guide says "any NumPy
dtype," which is false.

The GPU documentation says `uint16`, `uint32`, and `uint64` have no PyTorch
counterpart. In the tested PyTorch 2.10 runtime, `torch.uint16`, `torch.uint32`,
and `torch.uint64` all exist. Operations on those dtypes may be limited across
supported PyTorch versions, but the current explanation is no longer accurate.
A capability table keyed to the installed backend is preferable to a permanent
hard-coded set.

Other missing data features that potential adopters commonly need:

- variable-sized messages or a valid-length field;
- resize/reconfigure/version negotiation;
- strides/layout metadata and non-contiguous inputs without forced conversion;
- read-only consumer handles;
- a public zero-copy writable view/context (the implementation has `_array` and
  a locked unsafe read, but no purpose-built writer view);
- batched writes or atomic transactions across multiple streams;
- timestamps from a monotonic clock and user-defined metadata;
- a producer heartbeat and stale-data age policy.

## Discovery and lifecycle gaps

Names and introspection are convenient on Linux, but `list_streams()` and
`purge()` return no useful discovery on macOS or Windows. There is no namespace,
owner/application tag, lease, TTL, or scoped cleanup. All users of a process
must also agree on `PYSHMEM_LOCK_DIR`; otherwise they can access the same data
while taking different locks.

GPU lifecycle is intrinsically less persistent than CPU lifecycle. PyTorch's
CUDA IPC rules require the sending process to remain alive while consumers hold
the tensor. The repository partly documents fallback behavior, but the top-level
"persistent streams" message needs to distinguish CPU segments, mirrored
payloads, CUDA handles, and CUDA allocations.

## Usability gaps

- `read_new(safe=False)` is exposed but practically unusable. Without a held
  lock it raises; with a held lock it waits while preventing a conforming writer
  from producing the update.
- `read(out=...)` calls itself "zero-copy" in a docstring although it performs
  `np.copyto`; it is zero-allocation, not zero-copy.
- Implicit dtype conversion on writes can silently truncate or wrap values.
  There is no `casting=` policy.
- Poll intervals and timeout values are weakly validated.
- There is no high-level producer/consumer example demonstrating missed writes,
  crash behavior, and correct shutdown—the cases that determine whether a
  transport is safe in a real application.

## Features that would materially improve adoption

1. A capacity-N ring buffer with overrun counts and block/drop policies.
2. Waitable notifications (Linux futex/eventfd, Windows event, portable
   semaphore fallback) instead of micro-sleeps.
3. Explicit semantic profiles: `latest_value`, `queue`, and perhaps
   immutable/snapshot.
4. A backend-neutral tensor interface using DLPack/array protocols, while
   retaining optimized PyTorch adapters.
5. Namespaces, read-only opens, creator identity/heartbeat, and scoped cleanup.
6. A stable on-memory format specification and backward-compatibility policy.
7. GPU `out=` support, asynchronous APIs accepting a CUDA stream/event, and
   pinned-host mirror support.
