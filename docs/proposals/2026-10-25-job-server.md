# Proposal: a job server on RunPod workers

Date: 2026-10-25. Status: proposal, nothing implemented. Phase 1 of ICO
item 31 ("a pool across jobs"): the resource pool of the job's main
process survives between jobs on a warm worker. Phase 2 (MP workers that
outlive a run, with their pools) is discussed after phase 1 is measured;
see section 9. Every item of section 8 is a separate PR, as `CLAUDE.md`
requires.

## 1. Summary

A warm RunPod worker loads every model again on every job: about 115 s of
a 471 s `clean_video` run of a 30 s video. The pool is fine; its process
is not: the handler starts a new runner process per job, and the pool
dies with it.

The proposal changes one principle of `docs/design/remote.md`: **one job,
one new process** becomes **one code fingerprint, one process**. The
handler keeps a long-lived runner, the **job server**
(`python -m ico.remote --serve`), and sends it jobs one at a time over
the protocol of today (a job in, JSON events out). The server stays a
separate process, so a crash or a kill never takes the handler down; it
exits, and the next job starts a new one, whenever reuse is in doubt: a
different code fingerprint, a job that ended with `run_error` or
`load_error`, a cancel that did not stop in time, or `max_jobs`.

Expected gain: ~117 s per video from the second job on a warm worker
(471 → ~354 s, −25%, ~$0.11 per video on an RTX PRO 6000 at $3.49/h),
when jobs are queued or arrive within the idle timeout. Effort: about
two days for the server and the handler, half a day for GPU accounting
in the pool.

## 2. Evidence

Domain `apriori3d/ico-mcp-video` (main 31629d9), endpoint
`474nahcg47n9dl`, ico-mcp 0.1.1, apriori-ico 0.1.0. Two `clean_video`
runs back to back on the same warm worker (`67r2m9uwgtmwwp`, RTX PRO
6000, `idleTimeout` 180 s). Both profiles show `loads: 1` for every
resource, with the full load time:

| Resource | Load per job | Process |
|---|---|---|
| `diffueraser@1` | 91–95 s | job runner |
| `diffusion.trt@1` (UNet) | 14 s | job runner |
| `diffusion.trt@1` (BrushNet) | 7.5–8 s | job runner |
| `vae.trt@1` | ~2 s | job runner |
| `paddle.ocr@1` | 2–6 s | `ocr worker` (MPWorker) |
| `transnetv2@1` | 0.8 s | `split worker` (MPWorker) |

`clean_video` has two `worker` nodes, `split worker` (around
`split_master`) and `ocr worker` (around `scan_scenes`);
`clean_scenes_auto`, where DiffuEraser runs, has none. So ~117 s of the
~120 s are loads in the main process of the job, which phase 1 keeps.

## 3. Where a job's process and pool live today

```
RunPod worker (container, lives until idleTimeout or a release)
└─ handler.py → ico.remote.runpod_handler.handler   (lives with the worker)
   └─ python -m ico.remote --workdir <volume>/ico-jobs/<job id>   (one per job)
      │  pool() of this process: diffueraser, diffusion.trt ×2, vae.trt
      ├─ MPWorker "split worker" runner (spawn; one per run): transnetv2
      └─ MPWorker "ocr worker"   runner (spawn; one per run): paddle.ocr
```

- `runpod_handler.handler` starts `python -m ico.remote --workdir DIR`
  for every RunPod job, writes the job to its stdin, closes stdin, and
  yields its stdout lines until the process exits. On RunPod's cancel it
  sends SIGTERM, then SIGKILL after `CANCEL_GRACE` (10 s).
- `ico.remote.__main__` reads one job and calls `runner.serve`, which
  runs it and returns: the process, its `pool()`, the imported operator
  modules and the registry end with the job.
- `runner._load` writes the job's `sources` under `<workdir>/code` and
  imports the packages by name from there (`load_operators`). The path is
  per job id, so the same code arrives at a new path every job.
- `spec.run` activates the runtime and always deactivates it in
  `finally`; `MPWorker` spawns its runner process on activation and joins
  it on deactivation. Phase 1 leaves this unchanged.

The local path (ico-mcp on cube) has the same shape for its own reasons:
`ico_mcp.jobs` starts a kernel process per run, so a run is cancelled by
killing it and a reload never cuts a run short. Phase 1 does not change
it (section 7).

## 4. Options

- **A. A job server per RunPod worker (recommended).** A long-lived
  `python -m ico.remote --serve` that runs jobs one after another; the
  handler owns it. Keeps the main pool, the imported modules and the CUDA
  context; isolation from the handler stays.
- **B. Run jobs in the handler's process.** Rejected: a crash or a CUDA
  fault kills the RunPod worker's handler, and a job that does not stop
  cannot be killed without it.
- **C. Several videos in one job.** Works today (a `stream` over the
  inputs with a `pipeline_ref` to `clean_video`): models load once per
  job. Kept for batches, but results come only at the end, one bad video
  needs `on_error`, and interactive runs (one video at a time from
  ico-mcp) do not benefit.
- **D. Faster loads.** Independent of A. `diffueraser` takes 91–95 s on
  the second job too, with its files in the page cache, so the time is
  probably CPU (building the pipeline, fusing the LoRA), not reading. A
  cache of fused weights would also help cold starts. Not part of this
  proposal; worth a profile of `DiffuEraser.load` on its own.
- **E. A + MP workers kept between jobs.** Phase 2 (section 9).

## 5. Design (option A)

### 5.1 The server

`python -m ico.remote --serve` (no `--workdir`; each job carries its
id):

- Reads requests from stdin, one JSON line each:
  `{"job": <job>, "workdir": "<path>"}`. Runs them one at a time with the
  code of today (`run_job` / `fetch_job`, the work directory lock,
  staging in and out). Writes the job's events to stdout as JSON lines,
  ending with exactly one `result` or `error`, as today. Between jobs it
  writes nothing and waits for the next line.
- **Code is loaded once.** The first run job loads its packages
  (`_load`) and the server records the job's **code fingerprint**: a
  hash of `sources` and `packages` (path, status, modules). Packages
  without sources come from the image (`ICO_CODE`), which is fixed for
  the process. A later job with the same fingerprint skips `_load`: its
  modules are already imported by name and registered. Its sources are
  still written to its own `<workdir>/code`, so a resumed job and
  `ICO_ASSETS_MANIFEST` (set per job) work as today. A job with another
  fingerprint is refused with `code_changed` (the handler never sends
  one, 5.2). Prefetch jobs carry no code and run in any server.
- **Per-job state is reset.** Environment variables a job sets
  (`ICO_ASSETS_MANIFEST`) are restored after it. The pool, the registry,
  the bucket mounts and imported modules stay.
- **Cancel.** SIGTERM during a job cancels that job (the token of
  today); the job stops between items, its result says `cancelled`, and
  the server waits for the next job. SIGTERM while idle exits.
- **Exit after the job** (the handler starts a new server for the next
  one) when the job ended with `run_error` (an exception in the flow:
  CUDA may be left in an error state, operators may hold half-built
  state) or `load_error` (modules half imported), or after `max_jobs`
  jobs (`ICO_SERVE_MAX_JOBS`, unlimited by default; a valve if leaks show
  up). `spec_error`, `missing_assets`, `busy` and `cancelled` keep it.

`python -m ico.remote --workdir DIR` (one job, then exit) stays for
`LocalExecutor` and tests.

### 5.2 The handler

`runpod_handler` keeps one server per worker (a module global):

- computes the job's fingerprint; if a server is running with another
  one, closes its stdin, waits for it to exit (it is idle), and starts a
  new one;
- writes the request line and yields the events up to the final one,
  then returns, leaving the server running;
- on RunPod's cancel (and the execution timeout, which RunPod delivers
  the same way): SIGTERM, then SIGKILL after `CANCEL_GRACE`. A killed
  server is gone; the next job starts a new one (cold);
- a server that exits without a final event (a crash, the OOM killer) is
  the `crashed` error of today; the next job starts a new one;
- jobs are serial: RunPod's default concurrency per worker is 1; a lock
  in the handler keeps a higher `concurrency_modifier` from interleaving
  two jobs in one server.

Nothing changes in the job, the events, `RunPod` (the executor) or
ico-mcp.

### 5.3 GPU memory across jobs

The pool budgets what it can move. A TensorRT engine
(`diffusion.trt`, `vae.trt`) is built on the GPU and cannot leave it:
its `to()` returns the object, so with the default GPU budget (0) the
pool "moves it back to RAM" on release and counts its ~6 GB per frame
size against the RAM budget while it stays on the GPU. Within one job
(one frame size) this is harmless; across jobs, engines of every frame
size the worker has seen pile up on the GPU, uncounted.

Proposal (core, its own PR): a resource class may declare that its
objects live on one device, e.g. `resident: ClassVar[str | None] =
"cuda"`. The pool then counts them on that device, never moves them, and
when the device holds more than `ICO_RESOURCES_GPU` it drops idle ones
(`close`), least recently used first, instead of moving them back. With
`ICO_RESOURCES_GPU` unset, resident objects are never dropped (today's
behavior, no regression within a job). On the endpoint,
`ICO_RESOURCES_GPU` (e.g. `32GB` on a 96 GB PRO 6000) then bounds
engines across frame sizes, and lets `diffueraser` stay on the GPU
between uses instead of moving back to RAM. The domain sets `resident`
on its two TRT resources and frees GPU memory in their `close`.

### 5.4 RunPod lifecycle

- **Idle timeout.** The pool lives as long as the worker. A job that
  arrives after `idleTimeout` gets a new worker and loads everything. The
  gain applies to jobs that are queued (RunPod hands a worker its next
  job at once) or arrive within the timeout; keeping a worker idle for
  180 s costs ~$0.17, more than the ~$0.11 the reuse saves, so the timeout
  stays short for single interactive videos and the gain is for series.
- **Releases.** A new image (a release rollout) replaces the workers; a
  new server starts with it. The fingerprint covers what a job carries
  (draft code); the image covers the rest (ico, ico-mcp, the domain's
  stable packages, Python).
- **`refresh_worker`** must stay off in the handler (it stops the worker
  after each job).
- **Workdirs, resume and the lock** are per job, unchanged.

### 5.5 Profile

Profiles keep their per-run scope (the profiler attaches its pool
listener for the run). A warm job shows `hit` (or `move`) instead of
`load`, with `loads: 0`: the check that the server works.

## 6. Risks

- **Memory carried across jobs.** Caches in operators' modules, Python
  objects kept by a resource, the CUDA caching allocator's fragmentation.
  Mitigation: `run_error` restarts the server, `ICO_SERVE_MAX_JOBS` is a
  valve, the pool's budgets bound what it holds. To watch: the worker's
  RSS and GPU memory over a series of 10 jobs.
- **State in operators and resources.** Operator classes are built anew
  for every job (`cls(**params)`); what is shared is module-level state,
  registered function nodes, and the resource objects themselves (e.g.
  `routed` patches DiffuEraser's pipeline and restores it on exit). A
  test runs the same job twice in one server and compares the results;
  the domain's operators should not keep results in globals.
- **CUDA context reuse.** TensorRT engines are bound to the context, so
  reuse is only within one process, which is the point. A fault
  (illegal address) makes the context unusable: it surfaces as
  `run_error`, and the server exits.
- **A fingerprint that is too coarse.** Any edit of a draft module
  restarts the server and reloads every model. On purpose (old classes
  in `sys.modules` and pool keys made of `ref` and fields, not of the
  class, would otherwise serve stale code). Narrow it only if
  iterations on drafts make it hurt.
- **Paddle and torch.** Unchanged: Paddle loads only in the `ocr
  worker` process, which is still spawned per run in phase 1.

## 7. Local runs (cube)

ico-mcp's kernel per run reloads models for every local run too, by
design (cancel by kill, reloads that do not cut runs short). The pieces
of phase 1 (the serve loop, the fingerprint, the restart rules) are what
a warm run kernel in ico-mcp would need, but that is a change of
ico-mcp's run model and a decision of its own; not proposed here.

## 8. Plan of work

One PR each:

1. **Job server** (`apriori3d/ico`): `python -m ico.remote --serve`,
   the fingerprint, per-job environment reset, cancel and exit rules;
   `runpod_handler` keeps the server. Tests without network, through the
   handler's async generator with a fake domain package whose resource
   records its loads and the process id: two jobs with the same code →
   one load, one pid; other sources → a new pid; `run_error` → a new pid;
   a cancelled job → the same pid, and the next job runs; a job that
   ignores SIGTERM → killed, the next job runs in a new server; a crash →
   `crashed`. `docs/design/remote.md` ("Cold starts", the runner
   process) updated. ~1.5–2 days.
2. **Resident resources in the pool** (`apriori3d/ico`, 5.3), with
   tests of the budget and LRU drops. ~0.5 day.
3. **Domain and endpoint** (`ico-mcp-video`, by its owner): `resident =
   "cuda"` on `diffusion.trt` and `vae.trt`, `ICO_RESOURCES_GPU` on the
   endpoint, an image on the new apriori-ico. Measure two `clean_video`
   runs back to back on one warm worker: `loads: 0` and the time saved,
   then a series of 10 for memory.

## 9. Phase 2 (after phase 1 is measured)

MP workers that outlive a run (the design noted under item 31: a worker
parks its runner process on deactivation, keyed by worker type, name and
code fingerprint, and the next activation takes it and sends it a new
flow). For `clean_video` it would save Paddle's and TransNetV2's loads
(3–7 s) plus spawning two processes and importing torch and Paddle (not
measured); for pipelines whose models live in a `gpu` worker
(`animate_loop`, Wan2.2) it is the whole gain. It touches the worker
state machine, channels and shutdown (with the open issues of joining
runners that returned tensors), so it gets its own design.

## 10. Open questions

- Should `ICO_SERVE_MAX_JOBS` have a finite default (e.g. 50) until a
  series has shown memory is stable?
- Is `cancelled` a reason to restart (a run stopped between items is
  clean, but an operator interrupted in a long call may not be)?
  Proposed: keep the server, restart only when SIGTERM was not enough.
