# Performance critique

## Implementation update

Correct sequence checking was added to no-mirror GPU safe reads. After keeping
the CPU failure-handling path out of an extra context-manager hot path, the
128x128 CPU roundtrip measured 167,155 Hz, close to the original 169,346 Hz.
The now sequence-safe GPU roundtrip measured 27,516 Hz versus the original
31,035 Hz. That cost is accepted for correctness; CUDA event/stream-aware async
operations remain the next route to recovering throughput without weakening the
contract. A new spawned-process harness now supplies calibrated repetitions,
percentile latency, a raw shared-memory comparison, JSON output, and a checked-in
64 KiB result. A comparable PyTorch/CUDA process baseline remains open.

## Measured results on the supplied machine

The checked-in benchmark smoke suite produced:

| Case | Result |
|---|---:|
| CPU 128x128 write + safe read | 169,346 Hz (5.91 us/roundtrip) |
| GPU 128x128 NumPy write + safe read | 31,035 Hz (32.22 us/roundtrip) |
| CPU 1024 MVM pipeline | 9,539 Hz |
| GPU 1024 host-upload MVM pipeline | 23,494 Hz |
| GPU 1024 device-resident MVM pipeline | 29,313 Hz |

These are good smoke-test numbers. They show the implementation is not grossly
slow on its favored machine. They are not IPC latency numbers: writer and reader
objects live in the same process.

Targeted microbenchmarks separate library overhead from copying:

| Operation | 128x128 float32 | 1000x1000 float32 |
|---|---:|---:|
| CPU `write()` | 4.40 us | 61.47 us |
| CPU allocating safe `read()` | 1.30 us | 55.23 us |
| CPU `read(out=...)` | 1.23 us | 56.30 us |
| Raw `np.copyto` | 0.76 us | 54.89 us |
| pyshmem lock acquire/release only | 2.47 us | 2.44 us |
| GPU device `write()` | 10.14 us | 13.62 us |
| GPU safe clone `read()` | 7.04 us | 10.49 us |
| GPU device write + read | 17.11 us | 24.48 us |
| GPU NumPy `write()` | 26.39 us | 186.25 us |
| Raw GPU copy + sync | 6.35 us | 9.66 us |
| Raw GPU clone + sync | 6.51 us | 10.96 us |
| Raw copy+clone, one batched sync | 5.46 us | 12.31 us |

Numbers are single-run local observations, not universal claims. They identify
where to optimize.

## Optimizations left on the table

### 1. Every GPU operation globally synchronizes the device

GPU writes, reads, clears, and mirrored transfers originally called
`torch.cuda.synchronize(device=...)`, serializing unrelated work on the device.
**Resolved for synchronous operations:** pyshmem now records and waits on an
event in the active CUDA stream, and regression coverage forbids the global
synchronize API. Publication remains synchronous so another process cannot
observe an incomplete payload.

The remaining step is an asynchronous API with an IPC-capable event/publication
protocol. Returning an ordinary local event is insufficient because another
process also needs to know when the payload behind a published generation is
safe to consume.

### 2. NumPy-to-GPU writes take an avoidable path

`torch.as_tensor(value, device='cuda')` first creates/transfers a temporary GPU
tensor, then pyshmem copies that tensor into the shared tensor. That adds a D2D
copy and allocation/lifetime overhead. For a 4 MB input, pyshmem took 186.25 us;
a direct pageable host-to-destination copy took 165.96 us, and a pinned
host-to-destination copy took 143.43 us in the same environment.

**Direct destination copy resolved:** NumPy and CPU tensor inputs now stay on
the host until ``shared_cuda_tensor.copy_``, avoiding the temporary GPU tensor
and extra D2D copy. A 4 MB local follow-up measured median 171.70 us versus the
original 186.25 us. **Pinned staging is also resolved:**
``SharedMemory.pinned_buffer()`` reuses page-locked host storage and measured a
149.67 us median for the same 4 MB write. The remaining related issue affects
GPU-to-CPU mirroring, which currently creates a CPU result and then copies it
again into shared memory.

### 3. Polling is costly and has scheduler-dependent latency

`read_new()` and odd-sequence waits use Python loops plus `time.sleep(1e-5)` or
`time.sleep(1e-6)`. These requests are below typical scheduling granularity,
burn wakeups under load, and provide unpredictable tail latency. The async form
still polls; it merely yields through `asyncio.sleep`.

Use a generation counter plus futex/event/semaphore notification. Retain a spin
phase only as an opt-in low-latency policy, ideally adaptive and measured.

### 4. The mandatory file-lock path dominates small CPU writes

At 128x128, raw copy cost was 0.76 us, lock acquire/release was 2.47 us, and a
full write was 4.40 us. File locking is the majority of controllable overhead.
A native process-shared mutex or atomic single-writer fast path could materially
improve small-message rates. Make `single_writer=True` an explicit construction
policy rather than silently weakening safety.

### 5. Safe reads necessarily copy, but the API can make choices clearer

For a 4 MB payload, pyshmem safe-read performance is essentially memory-copy
bandwidth. `out=` removes allocation but cannot remove the copy, and measured
latency was nearly identical. The docs should not call it zero-copy.

Double buffering can give a stable zero-copy read lease: a writer fills the
inactive slot and atomically publishes it; readers pin a generation/slot until
done. That would also solve crashed-write recovery and strengthen GPU semantics.

### 6. Metadata and Python property traffic add hot-path overhead

Each operation repeatedly converts NumPy float metadata to Python values,
updates timestamps with `time.time()`, checks Python state, and enters context
managers. This is acceptable for large arrays but visible for tiny control
vectors. Fixed integer metadata and a compact native hot path would reduce both
latency and memory-model ambiguity.

## Problems in the published benchmark story

### Implementation update

Items 1–3 below are resolved for CPU by ``benchmarks/benchmark_ipc.py``: it
crosses a spawn process boundary, compares an explicitly unsafe raw shared-memory
lower bound, calibrates duration, repeats measurements, and records p50/p95/p99
latency in versioned JSON. The historical criticisms are retained to explain
why the original smoke figures must not be presented as IPC claims. GPU still
needs a CUDA/PyTorch process baseline and event-aware measurements.

1. **No process boundary.** All timed writer/reader handles are in one process.
   This omits scheduling, wakeup, consumer startup, CUDA context, and
   cross-process contention costs—the reasons adopters choose IPC.
2. **No baseline.** There is no raw `multiprocessing.shared_memory`, Pipe/Queue,
   PyTorch queue/CUDA IPC, or direct copy baseline.
3. **No latency distribution.** Only a single aggregate rate is printed. There
   are no repetitions, confidence intervals, p50/p95/p99, cold attach, or
   contention results.
4. **Public size tables are not reproducible from a checked-in command.** The
   test file has fixed default shapes and no parameterized 100/1000/10000 image
   sweep or artifact containing raw invocations/results.
5. **Duration claim conflicts with defaults.** README says every timed case ran
   at least 1.5 seconds. The observed default timed regions were roughly 8-64
   ms, and the code has no minimum-duration calibration loop.
6. **MVM benchmarks copy the supposedly resident matrix each iteration.** Both
   CPU and GPU call `matrix_reader.read()` inside the timed loop; safe reads copy
   or clone the full matrix. The benchmark therefore combines transport,
   allocation/copy, vector creation, synchronization, and matmul. It is not a
   clean measure of keeping a matrix resident in shared memory.
7. **CI does not enforce performance.** `pyshmem_ENFORCE_BENCHMARK` is explicitly
   zero, so benchmark regressions cannot fail CI.

Create a standalone benchmark harness that pins CPU affinity where appropriate,
uses spawned writer/reader processes, calibrates duration, emits JSON with full
environment data, compares baselines, and checks statistically defensible
regression thresholds on dedicated hardware.
