# Alternatives and competitive position

## Implementation update

The reliability fixes improve pyshmem's position as a small named latest-tensor
exchange, but they do not change the market comparison below. There is still no
exact drop-in alternative, while queueing, distributed operation, multi-GPU
collectives, and full pipeline orchestration remain better served by the listed
specialized systems.

There is no obvious exact drop-in that combines pyshmem's tiny named CPU API
with PyTorch CUDA IPC. That is the project's real niche. Alternatives become
more complete when the requirement is narrowed to CPU objects, queues, ML
process topology, or serious multi-GPU communication.

| Alternative | Stronger than pyshmem when... | Weaker than pyshmem when... |
|---|---|---|
| Python `multiprocessing.shared_memory` | CPU-only users want a standard-library primitive, `SharedMemoryManager`, and Python 3.13's public `track=False` lifecycle control | Users must build dtype/shape metadata, locking, discovery, and snapshots themselves |
| `torch.multiprocessing` Queue/Pipe | The workload already uses PyTorch and needs real queueing/backpressure and supported tensor transfer patterns | Named late attachment and a tiny NumPy-first API matter |
| Ray object store | Users need immutable CPU objects, zero-copy NumPy reads on a node, ownership, spilling, scheduling, and multi-node operation | Deployment weight and latency matter; direct mutable GPU slot semantics are desired |
| CuPy CUDA IPC/runtime APIs | Users need lower-level CUDA handle control, CuPy interoperability, and explicit event/stream design | They want lifecycle/metadata/locking handled and use PyTorch rather than CuPy |
| `torch.distributed` / NCCL | Users need collectives, multi-node/multi-GPU communication, groups, and established distributed-training semantics | They need arbitrary named late joiners and a latest-frame shared slot |
| NVIDIA Holoscan/UCX | Sensor/medical pipelines need a full operator graph, CPU/GPU tensors, DLPack interoperability, scheduling, and distributed transports | A small dependency and standalone named memory primitive are priorities |
| NVIDIA NVSHMEM | HPC users need symmetric GPU memory, GPU-initiated put/get, atomics, collectives, CUDA-stream ordering, and multi-GPU/cluster operation | Python ergonomics, NumPy support, and no HPC runtime setup matter |
| A purpose-built shared ring buffer | Users need capacity, ordered delivery, overrun detection, and backpressure | They also need CUDA IPC and named metadata out of the box |

## Implications for positioning

pyshmem should not try to out-feature Ray, Holoscan, NCCL, or NVSHMEM. It can win
by being the well-tested, low-dependency **named latest-tensor exchange for one
Linux host**. That positioning needs:

- honest single-slot semantics;
- impeccable failure recovery and lifecycle behavior;
- explicit consistency levels;
- genuinely reproducible low-latency benchmarks against raw shared memory and
  PyTorch multiprocessing;
- optional capacity-N buffering for users who need a real stream;
- DLPack or array-protocol interop so PyTorch is an adapter, not the entire GPU
  identity.

The name also risks expectation collision with OpenSHMEM/NVSHMEM concepts.
Those systems provide a partitioned global address space, remote put/get,
atomics, and multi-GPU/node communication. pyshmem is local named IPC and should
say so immediately in its tagline.

## Primary references checked

- Python shared memory and `track` behavior:
  <https://docs.python.org/3/library/multiprocessing.shared_memory.html>
- PyTorch CUDA and multiprocessing guidance:
  <https://docs.pytorch.org/docs/stable/notes/multiprocessing.html>
- PyTorch CUDA semantics, streams, pinned memory, and async copies:
  <https://docs.pytorch.org/docs/stable/notes/cuda.html>
- Ray zero-copy NumPy/object-store behavior:
  <https://docs.ray.io/en/latest/ray-core/objects/serialization.html>
- CuPy CUDA IPC handle API:
  <https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.runtime.ipcOpenMemHandle.html>
- NVIDIA Holoscan distributed CPU/GPU tensor transport:
  <https://docs.nvidia.com/holoscan/sdk-user-guide/using-the-sdk/create-a-distributed-application>
- NVIDIA NVSHMEM API and memory model:
  <https://docs.nvidia.com/nvshmem/api/>
- Current pyshmem PyPI metadata/provenance:
  <https://pypi.org/project/pyshmem/>
- Current public repository and issue tracker:
  <https://github.com/jacotay7/pyshmem>
