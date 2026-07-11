# Prioritized recommendations

## Progress summary

Detailed evidence is in [Implementation status](00-implementation-status.md).

- P0.1: **implemented with explicit invalid-state recovery and bounded reads**;
  double buffering remains a possible stronger future design.
- P0.2: **implemented**.
- P0.3: **implemented** for descriptor lifetime; fork-state hardening remains.
- P0.4: **implemented** for ownership validation and CUDA cleanup scope;
  metadata magic, namespaces, and dry-run remain.
- P0.5: **implemented at the generation protocol level**; cross-process CUDA
  events and asynchronous stream ordering remain.
- P1.5: **partially implemented** with failed-copy, writer-kill, timeout,
  descriptor, purge, and GPU odd-sequence tests.
- P1.6: **repository documentation corrected**; hosted docs still need rebuild.
- Repository hygiene: Python 3.9 CI, complete sdist tests, and obsolete artifact
  removal are **implemented**.
- P1.2: **format foundation implemented** with a documented v3 magic/header,
  fixed-width aligned fields, explicit little-endian encoding, feature flags,
  and v2 attachment compatibility. Strict semantic corruption and payload
  geometry validation is also **implemented** for open, discovery, and purge.
  The interprocess **memory model is now specified** (encoding, alignment,
  seqlock protocol, reliance assumptions, and the absence of hardware barriers),
  with correctness claims **narrowed** to validated architectures (x86-64,
  aarch64) and the hot-path counter alignment enforced by a regression test. A
  native acquire/release atomic backend remains the open enforcement step.

The original prioritized list below is retained so unfinished portions remain
visible rather than disappearing when an item is partially completed.

## P0: fix before describing the package as production-stable

1. **Make publication crash-safe.** Prefer double-buffering and atomic slot
   publication. At minimum, define invalid/recovery state, detect dead/stale odd
   generations, and give reads a bounded failure mode.
2. **Honor one deadline across thread and process locks.** Add a regression test
   where another local thread holds the lock longer than the requested timeout.
3. **Reference-count and close per-name lock state.** Add a test asserting that
   thousands of unique create/close/unlink cycles leave descriptor count
   bounded. Clear fork-inherited state safely.
4. **Scope purge to objects proven to belong to pyshmem.** Add metadata magic,
   namespace/application IDs, ownership validation, `--dry-run`, and explicit
   confirmation. Remove global PyTorch orphan sweeping from ordinary purge.
5. **Resolve GPU consistency semantics.** Either implement cross-process CUDA
   event/generation synchronization or explicitly make no-mirror reads
   `consistency='best_effort'`. `safe=True` must never return while the protocol
   says a write is active.

## P1: needed for a credible 1.x adoption story

1. Rename/position the existing primitive as `LatestValue`/`SharedTensor`, or
   implement capacity-N ring-buffer streams with missed-item reporting.
2. Publish a memory-format and compatibility specification using fixed-width
   integer fields, a magic number, endianness, feature bits, and atomic ordering.
3. Replace or isolate private CPython/PyTorch dependencies; use public
   `track=False` on Python 3.13+.
4. Add scheduled and release-blocking GPU CI on at least one supported CUDA/
   PyTorch combination. Test the minimum and newest supported PyTorch versions.
5. Add fault-injection tests: kill writer mid-copy, CUDA error after odd publish,
   malformed/truncated metadata, unlink/recreate with live handles, fork/spawn,
   and prolonged multi-reader/multi-writer stress.
6. Correct the README contradiction, rebuild hosted docs, narrow platform claims,
   and document trusted-process assumptions around pickle.
7. Replace public benchmark tables with a checked-in harness and versioned JSON
   results, cross-process baselines, repetitions, percentiles, and dedicated
   performance hardware.

## P2: performance and ecosystem expansion

1. Add CUDA-stream/event-aware asynchronous reads and writes; avoid whole-device
   synchronization.
2. Directly copy NumPy/CPU tensors to shared GPU storage; offer reusable pinned
   staging/mirror buffers.
3. Add waitable notifications with a portable fallback and an opt-in adaptive
   spin policy.
4. Add GPU `out=` and explicit `casting=`; reject unsupported parameter
   combinations instead of silently ignoring them.
5. Add DLPack/array-interface adapters and capability-driven dtype support.
6. Add namespaces, read-only consumers, heartbeat/staleness metadata, and a
   scoped registry.
7. Split the monolith by format, backend, synchronization, and lifecycle.

## Repository hygiene checklist

- Enable public issue creation and add issue templates.
- Add SECURITY, CONTRIBUTING, support/compatibility policy, and changelog.
- Test Python 3.9 or raise the declared minimum to 3.10.
- Add branch coverage, type checking, dependency updates/auditing, and stress CI.
- Gate PyPI publishing on a tested immutable build artifact.
- Remove tracked `dist-release/pyshare-1.0.0*` artifacts.
- Include `tests/conftest.py` in the sdist or omit tests entirely from it.
- Derive package/docs version from one source.
- Stop ignoring the improvement plan if it is intended as project guidance.
- Decide whether `shmpipeline`'s private coupling is supported; test and version
  that contract if it is.

## Suggested release framing

Until P0 items are fixed, a more defensible classifier is Beta, with scope stated
as:

> Named, fixed-shape latest-value exchange for NumPy arrays and PyTorch CUDA
> tensors between trusted processes on one Linux host. CPU and mirrored GPU
> modes provide sequence-checked host snapshots; no-mirror GPU reads prioritize
> throughput and require a live producer.

That narrower claim is still useful and differentiated. Reliability improvements
can then justify restoring `Production/Stable` without requiring the project to
become a full distributed framework.
