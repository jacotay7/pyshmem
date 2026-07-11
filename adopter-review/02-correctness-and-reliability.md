# Correctness and reliability critique

## Implementation update

The first remediation batch completed the four P0 items and the GPU sequence
item described below:

- failed or crashed writes now become an explicit negative invalid generation;
  reads raise `InconsistentStreamError`, and a later complete write repairs it;
- `read(timeout=...)` bounds active-writer waits;
- lock deadlines now cover local thread and process-file-lock acquisition;
- per-name lock state is reference-counted and its file is closed on final
  local handle close;
- purge validates name-to-hash ownership and global CUDA cleanup is opt-in;
- no-mirror GPU reads wait/recheck the sequence around clone+synchronize.

The original sections are retained as the evidence that motivated each change.
Since then the memory model has been specified (with narrowed platform claims),
the pickle trust boundary closed with a restricted unpickler, and the private
`resource_tracker` reach-in avoided on Python 3.13+. Publication ordering is now
enforced through x86-64 TSO, runtime `libatomic` acquire/release operations, or
a portable OS-lock barrier fallback. Reliance on torch's private reduction
internals and the CUDA lifecycle warning remain open.

## P0: a failed or crashed write can block readers forever

`_mark_write_started()` increments the shared sequence to odd. Copy and CUDA
synchronization happen next, and `_finish_write()` increments it to even. These
operations are not protected by `try/finally`, rollback, or crash recovery.

An injected `np.copyto` failure produced:

```text
first_error injected copy failure
sequence_after_failed_write 1 count 0
sequence_after_next_success 3 count 1 odd_means_stuck True
```

The next `read(safe=True)` spun until interrupted. A process killed between the
two sequence updates has the same effect. OS file locks recover after a crash,
but the payload protocol does not. Worse, the next writer begins from odd,
making the sequence even *during* its copy and odd after success—the exact
opposite of the intended invariant.

The fix needs more than a `finally` that declares partial data valid. Robust
choices include double-buffering with an atomically published active slot, or a
recovery protocol that marks the last payload invalid and lets the next writer
restore a known state while holding the lock. Safe reads also need a timeout or
producer-death/stale-write error instead of an infinite wait.

## P0: `acquire(timeout=...)` ignores local thread contention

`SharedMemory.acquire()` first calls `self._lock_state.thread_lock.acquire()`
without a timeout. Only after that succeeds does it use the requested timeout
for the cross-process file lock.

Probe result:

```text
{'requested_timeout_s': 0.05, 'result': 'acquired', 'elapsed_s': 0.501}
```

Applications cannot rely on the documented deadline. The implementation should
use one monotonic deadline across both lock layers and pass the remaining time
to each acquisition.

## P0: per-name lock state leaks an open file descriptor forever

`_THREAD_LOCKS` is a process-global dict keyed by lock path. Entries are never
evicted, and `_SharedLockState.file_handle` is never closed, even after every
handle for the stream is closed and the stream is unlinked.

Fresh-process probe:

```text
{'fd_before': 5, 'fd_after': 1206, 'lock_states': 1200,
 'gpu_open_locks': 0}
```

This is a direct failure mode for services that use unique job/frame/session
names. Use reference-counted per-name state, close and evict it when the last
local handle closes, and make fork handling explicit.

## P0: purge is broader than its name and documentation

On Linux, `purge()` selects every `/dev/shm/ps_*` entry. It does not first prove
that the metadata segment has pyshmem's version/layout/name relationship. An
unrelated application using the same common prefix can be deleted.

It additionally scans all `cuda.shm.*` objects and removes entries whose encoded
producer PID appears dead. Those files are PyTorch-wide, not pyshmem-specific.
Thus `pyshmem purge`, described as removing pyshmem state, can mutate leftovers
from unrelated PyTorch jobs owned by the same account. PID reuse also makes the
heuristic imperfect.

Cleanup should operate from a pyshmem-owned registry/namespace and validate a
magic value plus full segment relationships. Global PyTorch cleanup should be a
separate, explicitly dangerous administrator command, never an automatic part
of library-specific purge.

## P1: default GPU safe reads do not follow the consistency protocol

For `cpu_mirror=False`, `_read_consistent_gpu()` does only:

```python
result = self._gpu_tensor.clone()
torch.cuda.synchronize(device=self.gpu_device)
```

It neither waits for an even sequence nor rechecks it. A deterministic protocol
probe marked a write in progress and then called `read(safe=True)`:

```text
sequence_before_safe_read 1
safe_read_returned_while_odd (1024,) 1
```

A separate stress run alternated 256 MiB zero/one writes while another process
performed 120 reads and observed 0 mixed snapshots. That is useful negative
evidence: torn data was not reproduced on the RTX 5090. It does not establish a
portable guarantee, and the method demonstrably ignores the library's own
"write in progress" state.

For a true safe contract, coordinate CUDA streams across processes with IPC
events or another supported synchronization mechanism and recheck an atomic
generation. Otherwise rename the option/return contract so no-mirror reads are
explicitly best-effort.

## P1: the sequence protocol lacks a specified memory model

Metadata is a NumPy `float64` array in shared memory. Sequence/count mutations
are ordinary NumPy loads and stores, not interprocess atomic operations with
documented acquire/release ordering. A seqlock requires atomic counter access
and memory barriers so payload writes become visible before publishing the even
generation. Behavior that is plausible on x86-64 is not automatically correct
on ARM or every supported platform.

Use fixed-width integer metadata and a small native/FFI atomic layer, or a
well-defined synchronization primitive. Document byte order, alignment,
atomicity, and compatibility of the on-memory format.

## P1: unpickling a writable shared segment is a local code-execution boundary

**Resolved.** GPU attachment previously ran `pickle.loads(bytes(handle_shm.buf))`
on the writable mode-0600 segment, so any same-account process that could alter
it could make a later attacher unpickle arbitrary content. Reconstruction now
goes through `_RestrictedCudaUnpickler`, whose `find_class` resolves only torch's
known CUDA rebuild globals (`rebuild_cuda_tensor`, `Tensor`, `Size`,
`TypedStorage`) and inert dtype values; anything else raises `UnpicklingError`
before code runs. This is the review's "authenticate the handle and narrowly
reconstruct a known function" option, keeping torch's version-portable reduction
format. Validated on an RTX 5090 (legit round-trip plus a tampered-handle child
rejected with `disallowed global`) with a torch-independent CPU rejection test,
and documented in `docs/format.rst`. The remaining exposure is the set of
same-account processes that can write the segment, plus reliance on torch's
private `rebuild_cuda_tensor` internals.

## Additional reliability concerns

- Metadata read on `open()` validates the version but does not robustly validate
  every dimension, shape/size product, buffer length, boolean field, or stored
  name before constructing arrays.
- `_unregister()` and direct POSIX unlink rely on private CPython internals.
  Python 3.13 added the public `track=False` option; version-specific use of that
  API would reduce resource-tracker fragility.
- Torch internals such as `torch.cuda._lazy_init()` and serialized reduction
  implementation details can change without the compatibility guarantees of a
  public storage-format API.
- GPU tests consistently emitted: `Producer process has been terminated before
  all shared CUDA tensors released.` Passing tests with a lifecycle warning
  should not be the expected clean test outcome.
- `close()` suppresses segment-close exceptions broadly, hiding cleanup faults.
- SHA-1 truncated to 14 hex characters gives only a 56-bit name space and has no
  collision detection against the stored original name.
- Removing a lock pathname while old handles still have its inode open can let
  a recreated pathname refer to a different lock object. **Resolved for the lock
  file:** `_SharedLockState` records the lock inode and rebinds a stale handle
  on each acquire when the pathname resolves to a new inode, so an
  unlink/recreate cycle reconverges on one shared lock. New streams retain that
  per-name lock inode and carry a random instance id; stale handle-level unlink
  raises `StaleStreamError` instead of deleting a replacement. Regression tests
  and `docs/platforms.rst` cover old data/metadata mapping isolation, stable
  lock identity, and recreation behavior.
