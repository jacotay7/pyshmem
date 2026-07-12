# pyshmem — Developer Reference

## Purpose

pyshmem is a unified CPU/GPU shared-memory library for Python. It wraps `multiprocessing.shared_memory` with a clean API for named, typed, array-shaped streams that work identically on CPU (NumPy) and GPU (CUDA/PyTorch). The library is intentionally minimal — the entire implementation lives in `src/pyshmem/_shared.py`.

## Core Concepts

### Streams
A **stream** is a named slot in shared memory with a fixed shape, dtype, and storage backend. It is the only primitive this library exposes. Streams persist across process exits (on POSIX) and can be attached by any process that knows the name.

### Internal segments
Each logical stream `name` maps to up to three POSIX shared-memory segments:
- **data segment** — the array payload (`ps_<sha1hash>`)
- **metadata segment** — a 256-byte structured header (v3; legacy v2 was `float64[32]`) followed by a 256-byte name region (`ps_<sha1hash>_meta`)
- **GPU handle segment** — torch `reduce_tensor()` payload (`(rebuild_fn, args)`, pickled) for cross-process GPU tensor reconstruction (`ps_<sha1hash>_gpu`)

Names are hashed (SHA-1, first 14 chars) to stay under the POSIX segment name limit while remaining collision-resistant. Because the hash is one-way, the original user-visible name is stored verbatim in the metadata segment's name region (UTF-8, null-padded, after the header block) so `list_streams()`/the CLI can report the friendly name. `METADATA_TOTAL_BYTES = METADATA_BYTES (256) + METADATA_NAME_MAX (256)`; discovery and purge ignore legacy/unrelated segments whose stored friendly name cannot be validated against the hash.

### Metadata layout (METADATA_INDEX_* constants)
New streams use metadata version 3: a fixed 256-byte, little-endian structured header with magic, version/header size, flags, fixed-width dtype/shape/lifecycle fields, aligned uint64 count and int64 write sequence, followed by the 256-byte user-visible name region. `_MetadataView` preserves the existing internal index interface and can also attach to legacy version 2 `float64[32]` metadata. See `docs/format.rst` for exact offsets and compatibility rules. Fixed-width aligned fields prepare for, but do not themselves provide, native acquire/release atomics.

### Locking model
- Cross-process: `portalocker` file locks in `/tmp/pyshmem-locks-<uid>/` (or `$PYSHMEM_LOCK_DIR`)
- Per-thread: `threading.RLock` (re-entrant within a thread)
- The `_lock_state(name)` function returns/creates a reference-counted `_SharedLockState` shared across all `SharedMemory` handles opened for the same name in the same process; the last handle evicts the entry and closes the file descriptor.
- Crash recovery: `portalocker` uses OS-level file locks that are released automatically on process exit.
- Unlink/recreate safety: `_SharedLockState` records the lock file's inode and rebinds a stale handle when the pathname resolves to a new inode, so a destroyed-and-recreated stream reconverges on one lock. Streams also carry a random 128-bit `instance_id`; handle-level `unlink()` rejects a newer replacement with `StaleStreamError`.
- Fork hardening: an `os.register_at_fork` child handler resets each inherited `_SharedLockState` (fresh RLock, cleared held flag, reopened private fd) and drops cached CUDA IPC tensors.

### Write sequence protocol
Writers bracket payloads with odd/even sequence numbers:
1. Increment `WRITE_SEQUENCE` (now odd → write in progress)
2. Copy payload
3. Increment `WRITE_SEQUENCE` again (now even → write complete)

Readers poll until `WRITE_SEQUENCE` is even (stable), snapshot the data, then verify the sequence didn't change. This lock-free consistency mechanism is in `_read_consistent_cpu()` and `_read_consistent_gpu()`.

### Frame-id publication token
The v3 metadata header carries a user `frame_id` uint64 (the slot that was `reserved` before; excluded from the header CRC, defaults to 0, reads 0 on legacy v2). Writers pass `frame_id=` to `write()`/`write_locked()`/`write_view()`/`write_view_locked()`; `_finish_write` stamps `self._pending_frame_id` **before** the releasing sequence store, so a reader observing a stable sequence sees the matching token. Omitting `frame_id` leaves the field unchanged; `_abort_write` clears the pending token. Read it back with the `SharedMemory.frame_id` property — for a torn-free value, read it inside the same lock scope that snapshots the payload. This lets consumers establish cross-stream frame identity (e.g. a synchronized multi-camera fan-in) rather than inferring it from publication counts.

### GPU IPC
GPU streams share tensors cross-process via torch's **official** reduction: the producer exports with `torch.multiprocessing.reductions.reduce_tensor()` (storing the pickled `(rebuild_fn, args)` in the GPU handle segment) and the consumer reconstructs with `rebuild_fn(*args)`. The consumer deserializes the segment through `_loads_cuda_handle()` / `_RestrictedCudaUnpickler`, whose `find_class` only permits torch's known CUDA rebuild globals (`_ALLOWED_CUDA_GLOBALS`) and inert dtype values, so a tampered 0600 segment raises `UnpicklingError` instead of executing arbitrary code. This is deliberate — calling `storage._share_cuda_()` / `_new_shared_cuda()` directly (the old approach) bypasses torch's `shared_cache` + IPC ref-counter bookkeeping, which **leaks the producer's GPU allocation for the process lifetime** because the counter never reaches zero. With `reduce_tensor`, the producer can reclaim memory via `torch.cuda.ipc_collect()` once consumers release.

GPU-memory lifecycle rules (see `SharedMemory.close`/`unlink`):
- **Consumer** handles drop their CUDA tensor on `close()` — this decrements the producer's IPC ref counter (required for the producer to ever reclaim).
- The **owner** keeps its tensor on `close()` (so the stream stays mappable/reopenable in-process) and only releases it on `unlink()`, which also calls `torch.cuda.ipc_collect()`.
- A per-process weakref cache (`_LOCAL_GPU_TENSORS`) avoids re-importing handles within the creator process.

`purge()` removes only segments whose stored name validates against their exact pyshmem hash. Global orphaned `cuda.shm.*` cleanup is opt-in with `purge(include_cuda_orphans=True)` / `pyshmem purge --include-cuda-orphans`, because that namespace is shared by all PyTorch applications under the OS account.

## Public API

```python
import pyshmem

# Create
shm = pyshmem.create("my_stream", shape=(100,), dtype="float32")
shm = pyshmem.create("my_gpu_stream", shape=(100,), dtype="float32",
                      gpu_device="cuda:0", cpu_mirror=False)
# opt-in kernel wakeups: writers wake parked read_new consumers via a futex
shm = pyshmem.create("frames", shape=(480, 640), notify=True)
# auto-unlink on context exit
shm = pyshmem.create("tmp", shape=(10,), auto_unlink=True)
with pyshmem.stream("tmp2", shape=(10,)) as shm:   # always auto-unlinks
    ...

# Attach — open() reconstructs the stream as created.  For a GPU stream it
# auto-attaches to the CUDA device recorded in metadata; no need to pass
# gpu_device. Passing gpu_device explicitly still works (and must match).
shm = pyshmem.open("my_stream")
shm = pyshmem.open("my_gpu_stream")                 # auto-attaches to its cuda:N
shm = pyshmem.open("my_gpu_stream", gpu_device="cuda:0")  # explicit (must match)
shm = pyshmem.open("my_gpu_stream", gpu_device=False)    # CPU-mirror only (no GPU attach; requires cpu_mirror=True)
shm = pyshmem.open("my_stream", readonly=True)           # consumer handle: mutating ops raise PermissionError

# Discover
pyshmem.list_streams()    # returns sorted list of user-visible stream names
pyshmem.gpu_available()   # True iff torch is importable and CUDA is available

# Use
shm.write(array)          # CPU: numpy array; GPU: numpy or CUDA tensor
data = shm.read()         # returns np.ndarray (CPU) or torch.Tensor (GPU)
data = shm.read(out=buf)  # zero-alloc: writes into pre-allocated buffer
data = shm.read_new(timeout=1.0)         # blocks until a new write arrives
data = await shm.read_new_async(timeout=1.0)  # asyncio-safe variant
count = shm.wait_for_count(after=last_count, timeout=1.0)  # level-triggered
data = shm.read_after(last_count, timeout=1.0)

# Locking (explicit)
shm.acquire(timeout=0.5)
shm.read(safe=False)           # zero-copy view — only valid inside lock
shm.write_locked(value)        # write without re-acquiring lock (shmpipeline fast path)
with shm.write_view() as output:  # zero-copy, exception-safe publication
    output[...] = value
shm.release()

# Context manager
with shm.locked():
    shm.read(safe=False)
    shm.write_locked(new_value)
with pyshmem.locked_many([input_stream, output_stream]):
    ...

# Metadata
pyshmem.stat("my_stream")   # metadata-only attach/reuse inspection
shm.describe()             # human-readable summary string
cfg = shm.to_config()      # dict: name/shape/dtype/gpu_device/cpu_mirror
shm2 = pyshmem.SharedMemory.create_from_config(cfg)

# Liveness / staleness (consumer-side, no heartbeat thread)
shm.age                    # seconds since last completed write (inf if never)
shm.is_stale(max_age)      # True if latest write older than max_age seconds
shm.producer_alive()       # best-effort single-host PID liveness of creator
shm.creator_pid            # PID that created the stream

# Framework interop (DLPack) — consistent snapshot, not a live view
np.from_dlpack(shm)        # CPU -> numpy; torch.from_dlpack(shm) GPU -> tensor
shm.__dlpack_device__()    # (kDLCPU,0) for CPU, attached CUDA device for GPU

# Lifecycle
shm.close()               # detach this handle; stream persists
shm.unlink()              # destroy the stream entirely
pyshmem.unlink("my_stream")
pyshmem.unlink_quiet("x") # unlink; no error if the stream is already gone
pyshmem.purge()           # remove all validated pyshmem segments (see CLI)
```

## Constants

| Name | Description |
|------|-------------|
| `GPU_SUPPORTED_DTYPES` | `frozenset` of NumPy dtypes the installed torch can map (see Supported dtypes) |
| `InconsistentStreamError` | Raised when a writer failed or exited before publishing a complete payload |
| `StaleStreamError` | Raised when a handle operates on a stream that was unlinked and recreated (instance-id mismatch) |

## CLI

```bash
pyshmem list                     # list user-visible names of all streams
pyshmem unlink my_stream         # destroy a stream by user-visible name
pyshmem unlink stream_a stream_b # destroy multiple streams
pyshmem purge                    # remove all validated pyshmem segments
pyshmem purge --include-cuda-orphans  # additionally sweep global dead-producer CUDA IPC files
```

## Key Implementation Details

- **`create()`** calls `SharedMemory._create()` which creates segments atomically (with cleanup on failure).
- **`open()`** calls `SharedMemory._open()` which reads metadata to reconstruct shape/dtype without needing the caller to know them.
- **`open()` device resolution** (`_resolve_open_target_device`): omitting `gpu_device` auto-attaches to the stored device (`METADATA_INDEX_DEVICE_INDEX`); if that can't attach it falls back to the CPU mirror when one exists, else raises. An explicit `gpu_device=` must match the stored device and raises (no fallback) if it can't attach. `gpu_device=False` skips CUDA entirely and reads the host mirror as NumPy (requires `cpu_mirror=True`, else `ValueError`).
- **Waitable notifications** (`create(..., notify=True)`, `METADATA_FLAG_NOTIFY`): opt-in per-stream. `_finish_write`/`_abort_write` call `_futex_wake` on the shared `write_sequence` word when `self._notify`; `read_new`/`read_new_async` park via `_futex_wait` (`_wait_for_publication`) instead of `time.sleep`. Non-private Linux futex keys on the physical page (works cross-process at different vaddrs); the word is the LE low 32 bits of `write_sequence`, which changes every publication, so the compare-and-block is race-free. Parked waits are capped at `_NOTIFY_MAX_PARK` (50 ms) so dead-writer detection still runs. `_FUTEX_AVAILABLE` is Linux + little-endian only; elsewhere it falls back to polling. `self._notify` is derived from the metadata flag in `__init__`, so **default streams (incl. shmpipeline's) hit only one `if self._notify` check and no syscall.**
- **`readonly=True` handles** (`open(..., readonly=True)`): a per-handle guard (`_ensure_writable`) makes every mutating operation raise `PermissionError` — `write`, `write_locked`, `clear`, `acquire`/`locked`, `pinned_buffer`, unsafe (`safe=False`) reads, and handle-level `unlink`. It does not protect the segment: other writable handles to the same stream (and the owner) still publish. `describe()` reports the `readonly` flag.
- **Resource tracker suppression**: segments are opened through `_attach_segment()`, which passes the public `track=False` on Python 3.13+ (so they are never registered) and otherwise falls back to constructing the segment and calling `_unregister()` (the private `resource_tracker.unregister` reach-in). Either way child process exits don't spuriously warn about leaked shared memory.
- **Platform**: POSIX only (Linux and macOS). Full Linux support; macOS works but GPU IPC is not tested there. Windows is **not supported** (no POSIX shared-memory persistence / process-shared locks) and is excluded from CI.

## Supported dtypes

CPU streams (`DTYPE_TABLE`, stable integer codes in the persistent format):
`int8 int16 int32 int64 uint8 uint16 uint32 uint64 float16 float32 float64 bool complex64 complex128`.

GPU streams: **capability-driven**, not a fixed list. `GPU_SUPPORTED_DTYPES` is
built at import time by mapping each `DTYPE_TABLE` entry to `getattr(torch, name)`,
so a dtype is GPU-usable iff the installed torch exposes it (e.g. older torch lacks
`uint16/uint32/uint64`; newer torch adds them). `create(gpu_device=...)`
with an unsupported dtype raises `ValueError` at construction time. Never
hard-code the GPU dtype set — read `GPU_SUPPORTED_DTYPES`.

## Project Structure

```
src/pyshmem/
  __init__.py     # public surface: create open unlink unlink_quiet stream purge
                  #   list_streams gpu_available SharedMemory GPU_SUPPORTED_DTYPES
                  #   InconsistentStreamError StaleStreamError __version__
  _shared.py      # entire implementation (~2.7k lines)
  _cli.py         # CLI entry point (list / unlink / purge)
tests/
  conftest.py               # shm_name fixture (auto-cleanup via uuid)
  test_cpu_api.py           # CPU stream tests (pytest.mark.cpu)
  test_gpu_api.py           # GPU stream tests (pytest.mark.gpu, skipif no CUDA)
  test_benchmark.py         # microbenchmark timing tests
  test_benchmark_harness.py # spawned-process IPC benchmark smoke test
benchmarks/       # benchmark_ipc.py + checked-in versioned JSON results
docs/             # Sphinx sources (format.rst is the authoritative on-disk format spec)
```

Other tracked docs: `README.md` (adopter landing page), `CHANGELOG.md`,
`CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`, `IMPROVEMENTS.md`.

## Running Tests

```bash
# All tests (CPU + GPU if CUDA available)
python -m pytest tests/ -q

# CPU only
python -m pytest tests/ -m cpu -q

# GPU only (requires CUDA)
python -m pytest tests/ -m gpu -q
```

## Lint

After any significant code change, run lint and auto-format before committing:

```bash
ruff check .           # must be clean (zero errors)
ruff format .          # auto-fixes formatting; re-run check after
```

The CI lint job enforces both `ruff check` (E/W/F rules, line-length 79) and `ruff format --check`. Fix all reported issues before pushing.

## Test Coverage Policy

**Every code change must be accompanied by tests.** This is non-negotiable.

- **New feature** → add at least one happy-path test and one error/edge-case test.
- **Bug fix** → add a regression test that fails on the unfixed code and passes after.
- **Correctness fix** (race condition, leak, etc.) → add a targeted test using
  `monkeypatch` or `threading` to reproduce the scenario.
- **Public API addition** → test all new parameters and return values, including
  `closed` state checks if the method calls `_ensure_open`.

Test placement:
- CPU-only behaviour → `tests/test_cpu_api.py` (marked `pytest.mark.cpu`)
- GPU-specific behaviour → `tests/test_gpu_api.py` (marked `pytest.mark.gpu`,
  `@pytest.mark.skipif(not CUDA_AVAILABLE, ...)`)
- Use the `shm_name` fixture for all stream names (guarantees cleanup on teardown).

The CPU suite must pass at all times:
```bash
python -m pytest tests/test_cpu_api.py -q   # must be green before any merge
```

## Package Info

- Package name on PyPI: `pyshmem` (v1.2.0)
- License: GPL-3.0-only
- Required deps: `numpy>=1.26,<3`, `portalocker>=3.1`
- Optional deps: `torch>=2.2` (GPU support)
- Python: 3.9–3.13
- GitHub: `https://github.com/jacotay7/pyshmem`

## Integration with shmpipeline

shmpipeline consumes only pyshmem's public stream API. Its worker runtime uses
sorted `locked()` scopes, safe GPU snapshots, `write_view_locked()` output
transactions, `wait_for_count()`/`read_after()` level-triggered waits, and
`unlink_quiet()` cleanup. Private segment names, CUDA caches, and publication
fields are implementation details and must not become an integration contract.
