# Baseten rollout plan: cloud retraining driven from the app

Status: draft, 2026-11-01. Nothing here is built yet.

## Goal

Today the retrain loop only works on one Mac. The app's **Retrain models** button posts labels to
`retrain.py serve` on `http://localhost:9100`. That Mac then collects labels, rebuilds `cache-phone`,
trains the 10 s and 5 s BandNets (on MPS or on Modal), compares them with the installed pair, and
installs them by rebuilding the macOS app.

The target: **any copy of the app (iOS or macOS) labels recordings, starts a retrain, reviews the
result and receives the approved models, and no developer Mac has to be involved.** Baseten runs
the training, and can optionally host a scoring endpoint used to evaluate candidate models.

## What exists today, and where each piece moves

| Piece | Today | Target |
|---|---|---|
| Labels | App history on the Mac, `.wmgmodel` exports, labels the app POSTs to `:9100` | Backend (Fly) stores labels per session and per clip |
| Session audio | Backend `/audio`, `/live` clips (already in the cloud) | Unchanged |
| Phone cache (`cache-phone`, 1.8 GB) | Built on the Mac by `phone_cache.py` | Built inside the Baseten job from backend audio |
| Clean cache (`cache-v3`, 2.6 GB) | Mac; uploaded to a Modal Volume | Object storage bucket, mounted into jobs through BDN |
| Training | `train_baseline.py` on MPS or Modal (`modal_app.py`, `cloud.py`) | Baseten Training Job, same code and pinned versions |
| Compare and gate | `retrain.compare()` / `regressions()` on the Mac | Same code, at the end of the job |
| Core ML export and parity | `install.sh` → `export_coreml.py`, macOS only | Export in the job; parity checked on the device |
| Install | Rebuild the app with the new `.mlmodelc` | App downloads the approved model over the air (OTA) |
| Retrain service (`/status`, `/retrain`, `/approve`…) | `retrain.py serve`, localhost | A small hosted controller with the same API; the app only changes `TRAINER_URL` |
| Personal heads | App → backend `/models/personalized` → `personalize merge` on the Mac | Merged and evaluated in the job |

Keeping the controller's HTTP API identical means `RetrainClient` and `retrain_section.dart` barely
change. `TRAINER_URL` in `app_config.dart` already makes the endpoint configurable.

## Baseten facts that shape the design

From docs.baseten.co (Training, Truss, Deployments):

- **Training Jobs** are defined in a `config.py` built from `TrainingProject`, `TrainingJob`, `Image`,
  `Compute` and `Runtime`, and submitted with `baseten train push --config config.py`. They can also
  be submitted from Python with `truss_train` `push()`, which returns the job id. The docs list no
  REST endpoint for submitting jobs, so whatever triggers retrains needs Python and the SDK.
- **GPUs: only H100 and H200** are listed for training. BandNet has about 0.4 M parameters, so one
  H100 is far more than it needs, but a full retrain should take minutes. `availability_model=SPOT`
  is available. The SDK lists `cpu_count`/`memory` and treats `accelerator` as optional, so a
  CPU-only job may work; that needs a test.
- **Code upload:** the job directory is packaged and uploaded. For more than about 1 GB, the docs say
  to use object storage. Both caches are over that limit, so they go to a bucket.
- **Data in:** `TrainingJob.weights=[WeightsSource(source="s3://…|r2://…|gs://…", mount_location=…)]`
  mounts data read-only through BDN before `start_commands` run. Bucket auth uses OIDC or a
  `CUSTOM_SECRET`.
- **Training cache:** `CacheConfig(enabled=True)` gives `$BT_PROJECT_CACHE_DIR`, which persists
  across jobs in a project. It is pinned to one cluster (`require_cache_affinity`), so it is fine for
  scratch data but should not be the record of anything.
- **Results out:** `CheckpointingConfig(enabled=True)`. The job writes to `$BT_CHECKPOINT_DIR`, and
  `baseten train checkpoint files --job-id …` returns presigned URLs that **expire after 7 days**.
  The job should copy `runs/<ts>/` to our own bucket as well.
- **Secrets:** `Runtime.environment_variables={"X": SecretReference(name="…")}`.
- **Logs:** `baseten train job logs --job-id … --tail`.
- **Inference (optional):** a Truss with `model/model.py` (`load`/`predict`) and `config.yaml`, pushed
  with `baseten model push`. **Environments** (for example staging and production) keep stable URLs,
  and promoting a deployment moves traffic without changing the URL. Deployments can scale to zero.
  The **Truss Push GitHub Action** (`basetenlabs/action-truss-push`) can validate a deployment on a PR
  with a predict request and promote it on merge.

## Phases

### Phase 0: Prerequisites and decisions (about a day)

1. Commit the Modal work that is currently uncommitted on `new_approach` (`modal_app.py`, `cloud.py`,
   the `retrain.py` trainer switch, `session_report.py`, `test_cloud.py`). The Baseten trainer is a
   sibling of the Modal trainer and reuses its structure.
2. Baseten workspace: create an API key, run `baseten auth login`, and store the key as a GitHub
   secret named `BASETEN_API_KEY`.
3. Create a bucket (R2 or S3), for example `wmg-train`, with this layout:
   ```
   caches/cache-v3/<fingerprint>/...      immutable; the path changes whenever the cache changes
   caches/cache-phone/<fingerprint>/...
   runs/<ts>/...                          model.pt, model.onnx, metrics.json, oof_*.csv, coreml/
   retrains/<retrain_id>/                 status.json, labels.json, comparison.json, log
   models/manifest.json                   which runs are installed (the OTA pointer)
   ```
   `cloud.fingerprint()` already gives the cache identity. Paths named after it make BDN mounts
   immutable and cacheable.
4. Make the decisions listed under "Decisions needed" below.

### Phase 1: Baseten trainer, same results as the Mac (2–3 days)

Replace only the training step. The Mac still does labels, the cache, compare and install. This
step exists to prove that Baseten gives the same metrics.

**Status (2026-11-03):** the code is written and unit-tested against fakes; no job has run on
Baseten yet. It differs from the outline below: there is no bucket. Caches ride along in the job
upload the first time a fingerprint is used and are kept in `$BT_PROJECT_CACHE_DIR`; the run
folder comes back through Baseten's checkpoint downloads. No secrets are needed beyond the
Baseten API key in `~/.trussrc`. A bucket (or BDN mounts) comes back if the caches outgrow the
5 GB upload limit. Setup steps are in the training README under "Training on Baseten".

- `training/baseten/config.py`
  - `Image`: a PyTorch CUDA base image, with the same pins as `modal_app.py` installed in `run.sh`
    (torch 2.14.1, numpy 2.5.4, pandas 3.0.6, pyarrow 26.0.0, scikit-learn 1.9.1, pyyaml, onnx).
  - `Compute`: 1×H100 on SPOT first. Try a CPU-only variant to compare cost and time.
  - `weights`: the two cache prefixes, mounted at `/data/cache-v3` and `/data/cache-phone`.
  - `Runtime`: `CheckpointingConfig(enabled=True)`. Env: `WMG_MODEL` (`10s`/`5s`), extra args, and
    the bucket secret.
  - `start_commands`: `./run.sh`, which runs
    `python -m wmg_train.train_baseline --cache /data/cache-v3 --phone-cache /data/cache-phone --out $BT_CHECKPOINT_DIR …`
    and then copies the run folder to `runs/<ts>/` in the bucket.
- `wmg_train/baseten_cloud.py`, mirroring `cloud.py`:
  - `sync_caches()`: upload to the bucket under the fingerprint path when it is missing.
  - `train_remote(name, extra_args, line)`: `push()` the job with that model's args, follow the
    logs, and download `runs/<ts>/` from the bucket.
  - Cancellation: the same `BaseException` → cancel pattern as `cloud.train_remote`.
- `retrain.py`: `--trainer baseten` / `WMG_TRAINER=baseten`, next to `local` and `modal`.
- Tests: `tests/test_baseten_cloud.py`, using fakes the way `test_cloud.py` does.
- **Acceptance:** train `10s` and `5s` on the same cache fingerprints locally (MPS) and on Baseten.
  `regressions(new, old)` must be empty in both directions within `TOLERANCE`, and the confound gate
  must stay inside `GATE_RANGE`. Record wall time and cost per run.

### Phase 2: Take the pipeline off the Mac (about 1–1.5 weeks)

1. **Labels go to the backend.** Add `POST/GET /labels` (body:
   `session_id`, `category`, optional `clip_index`) next to `/models/personalized`. The app writes
   labels there as the user makes them, not only when they press Retrain. `labels.collect_labels`
   gains a backend source; the existing conflict reporting stays.
2. **Build the phone cache in the job.** The image needs `wmg_features` for Linux x86_64:
   `maturin build --release -m features/pyproject.toml` in CI, with the wheel installed in `run.sh`.
   The feature spec hash must match the one bundled in the app, so one commit pins both. The job's
   first stage pulls the labelled sessions' audio from the backend (`/audio?session_id=…`,
   `/audio/{id}/stream`), runs `playlist label-session` / `phone_cache`, and publishes the new
   `cache-phone/<fingerprint>`. `cache-v3` stays mounted read-only.
3. **One job, the whole run:** labels → cache → train 10 s → train 5 s → compare. All of it writes to
   `retrains/<id>/status.json` in the same shape `Run` writes today. The balance guard (more than 80%
   of phone audio in one class) and `MIN_FREE_GB` still apply.
4. **Controller.** This is `retrain.py serve`, refactored so its state lives in the bucket instead of
   `training/retrain/`.
   - `/retrain`: `push()` a job.
   - `/status`: read the job's status from the bucket.
   - `/approve` and `/reject`: update `models/manifest.json`.
   - `/cancel`: stop the job.
   - `/relaunch`: no longer needed.
   - It needs auth: a shared app token at first, per-user tokens later.

   It is small, Python, and needs a stable URL; see decision 1.
5. App: set `TRAINER_URL` to the controller and drop the `build_app` option. `step_labels` no longer
   needs the Mac's app history, but keep reading it for a while so labels made before this change
   still count.
- **Acceptance:** on an iPhone build with the Mac off, label clips → Retrain → watch progress →
  review the comparison → approve/reject. History lands in `retrains/`.

### Phase 3: Deliver models without rebuilding the app (about 1 week)

Approval currently rebuilds the app. With the Mac out of the loop, the app has to fetch models itself.

- **In the job:** export Core ML with `export_coreml.py`. coremltools can convert on Linux. The
  parity check that runs a prediction needs macOS, so on Linux it falls back to checking ONNX
  Runtime against PyTorch, which `export_onnx` already supports. The job also writes the parity
  fixture (`parity_fixture.py`) next to the model.
- **Manifest:** `models/manifest.json` → `{10s: {run, url, sha256, feature_spec_hash}, 5s: {...}}`,
  served by the controller or backend.
- **`wmg_detector` plugin:** download, then `MLModel.compileModel(at:)`, then store the result in
  Application Support. The plugin already refuses to score when `feature_spec_hash` doesn't match;
  keep that check. Score the bundled parity fixture with the new model, and switch only if P(AI) is
  within 0.01 of the expected value. Otherwise stay on the previous model and report the failure.
- Keep the last good model for one-tap rollback. The bundled model is the floor.
- **Acceptance:** approving a run updates the models on a running iOS and macOS app without a
  rebuild. A corrupted or mismatched model is rejected and the app keeps scoring.

### Phase 4 (optional): Baseten scoring endpoint (3–5 days)

On-device Core ML remains the product path. A hosted scorer pays off for:

- **Evaluating candidates on real recordings.** Score every labelled backend session with the
  candidate and the installed model before the approve/reject decision. This is what
  `session_report.py` does by hand today (headline AUC 0.739 on 537 app clips).
- Shadow-scoring every new `/live` clip server-side, to find disagreements worth labelling.
- Clients without Core ML (Android, web) later on.

Design:
- A Truss with a CPU instance and ONNX Runtime. `load()` reads `model.onnx` for both models plus the
  `wmg_features` wheel. `predict()` takes PCM or a backend audio id and returns
  `p_ai, band_scores, windows` (the same contract as `WmgDetector.scorePcm16`).
- Environments: **staging** serves the candidate and **production** serves the approved pair.
  Approval promotes staging to production, so the endpoint follows the same model as the app.
- Autoscaling: allow scale to zero, `min_replicas=0`. It is batch and evaluation traffic.
- CI: the Truss Push Action validates on PRs that touch `inference/` with a fixed predict payload
  (the parity fixture). The controller pushes and promotes when a run is approved.

### Phase 5: Continuous refinement

- **Active learning:** have the app ask for labels where it is unsure. Good candidates are clips
  where the two models disagree by more than 0.1 and clips with a P(AI) between 0.3 and 0.7. This
  targets the current weak spot (AI caught 25 of 69 at 0.5 on app clips).
- **Personal heads:** run `personalize merge` in the job. Use the uploaded embeddings as extra
  training signal for `clip_head`, and report base vs personalised accuracy per device.
- **Scheduled retrains:** for example nightly once N new labels have arrived. Trigger them with a
  GitHub Actions cron that calls the controller, or with a timer in the controller.
- Write `session_report` output for every run to `retrains/<id>/report.md` and link it in the app.

## Decisions needed

1. **Where the controller runs.**
   - (a) A second small Fly app next to `wmg-spectro-api` (recommended). It is the same platform,
     stays on, costs about $5/month, and can call `truss_train.push()`.
   - (b) A Baseten CPU Truss with custom endpoints. Status polling would keep waking it.
   - (c) GitHub Actions via `workflow_dispatch`. It has no live status API, so the app would poll
     the bucket.
2. **Audio leaving the Mac.** Today the code says "Audio never leaves the Mac" for cloud training.
   The backend already stores the audio, but Phase 2 means Baseten processes it too. Confirm that
   this is acceptable for the label-owned and licensed tracks in `training_data/`. `cache-v3` is
   derived log-mels only, so it can stay that way.
3. **GPU vs CPU training.** Only H100/H200 are listed. Run one CPU-only test job in Phase 1, then
   pick based on cost per retrain.
4. **Modal.** Keep `--trainer modal` until Phase 1 passes, then delete it or keep it as a fallback.
5. **Who can approve.** Anyone with the app, or an admin role? This affects auth in Phase 2.

## Risks

| Risk | Mitigation |
|---|---|
| Metrics drift between MPS, Modal L4 and Baseten H100 | Same pins and seed. The Phase 1 acceptance run uses `TOLERANCE` |
| `wmg_features` built for Linux has a different `feature_spec_hash` from the iOS/macOS build | Build all slices from one commit in CI; the job fails on a hash mismatch |
| Presigned checkpoint URLs expire after 7 days | The job copies results to our bucket; the bucket is the source of truth |
| Training cache pinned to one cluster | Use it only for scratch; durable data lives in the bucket and BDN |
| A bad model reaches devices over the air | Hash and `feature_spec_hash` checks, on-device parity fixture, last-good rollback, bundled floor |
| Phone data drifts to one class as users label mostly human music | The 80% balance guard stays on in the cloud; Phase 5 prompts target AI clips |
| Cost of idle compute | Training jobs are per-run; the scorer scales to zero; the controller is a ~$5 Fly machine |

## Suggested order and first concrete step

Phase 0 → 1 first. They are low-risk and prove that Baseten can train these models. Phases 2 and 3
are the actual move to the cloud and should land together, since the app can't receive models
until Phase 3 is done. Phases 4 and 5 can follow in any order.

The first concrete task is `training/baseten/config.py` with `run.sh`, plus a smoke job that runs
`--epochs 1 --steps 5 --folds none` (the same job as `modal_app.py::smoke`) against the bucket-mounted
caches.
