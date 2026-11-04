# Environment variables

Every setting the project reads from the environment, where to set it, and whether it is a
secret. **Secrets never go in a tracked file.** They live in one of these places only:

| Where | Holds | In the repo? |
|---|---|---|
| `backend/.env` | ACRCloud key/secret for local dev | No — gitignored; `backend/.env.example` is the template |
| Fly secrets (`fly secrets set`) | ACRCloud key/secret in production | No |
| GitHub Actions secrets | `FLY_API_TOKEN` | No |
| `~/.trussrc` (`truss login`) | Baseten API key | No — in your home folder |
| `~/.modal.toml` (`modal setup`) | Modal token | No — in your home folder |

## Backend (`backend/`, Rust API)

Read by `backend/src/config.rs`. Locally from `backend/.env` (via docker compose `env_file`);
on Fly from `[env]` in `fly.toml` plus Fly secrets.

| Variable | Secret | Default | Purpose |
|---|---|---|---|
| `ACR_ACCESS_KEY` | **yes** | empty | ACRCloud access key. Blank disables identification. |
| `ACR_ACCESS_SECRET` | **yes** | empty | ACRCloud access secret. |
| `ACR_HOST` | no | empty | ACRCloud host, e.g. `identify-us-west-2.acrcloud.com`. |
| `ACR_MIN_SCORE` | no | `50` | Matches below this score (0–100) are `possible_match`. |
| `ACR_PROJECT_ID` | no | — | In `.env.example` but not read by the code yet. |
| `IDENTIFY_CONCURRENCY` | no | `2` | Parallel ACRCloud requests. |
| `PORT` | no | `8000` (`9000` in Docker/Fly) | Listen port. |
| `DATA_DIR` | no | `./data` (`/data` in Docker/Fly) | Uploads, caches, jobs. |
| `MAX_UPLOAD_MB` | no | `100` | Upload size limit. |
| `MAX_DURATION_SECS` | no | `1200` | Longest accepted audio. |
| `CORS_ORIGINS` | no | `*` | Comma-separated allowed origins. Tighten before public launch. |
| `LIVE_CLIP_SECS` | no | `30` | Clip length saved from `/live`. |
| `LIVE_MAX_MINUTES` | no | `60` | Longest `/live` session. |
| `RUST_LOG` | no | `info` | Log level. |
| `HOST_PORT` | no | `9000` | Host port docker compose publishes. |
| `FLY_API_TOKEN` | **yes** | — | GitHub Actions deploy (`.github/workflows/ci-deploy.yml`). Set as a repo/environment secret. |

## Frontend (`frontend/`, Flutter)

Passed as `--dart-define=NAME=value`. **These are compiled into the app binary and can be
read by anyone who has the app — never pass a secret this way.** None of the current ones are.

| Variable | Default | Purpose |
|---|---|---|
| `API_BASE_URL` | platform default | Backend URL (e.g. `http://<LAN-ip>:8010` on a phone). |
| `TRAINER_URL` | `http://localhost:9100` | Retrain service. |
| `ACCOUNT_BACKEND` | `fake` | `http` once the account/billing API is live. |
| `LIVE_IDENTIFY` | `true` | Send live audio for identification. |
| `LIVE_SAMPLE_RATE` | `32000` | Mic sample rate. |
| `DEV_TOOLS` | `false` | Show dev tools on handsets. |
| `LIVE_WINDOWS` | `3` | Integration test only (`live_mic_test.dart`). |

## Training (`training/`)

Training does not load a `.env` file. Export these in your shell, or copy
`training/.env.example` to `training/.env` (gitignored), fill it in, and run
`set -a; source .env; set +a` before training.

| Variable | Secret | Default | Purpose |
|---|---|---|---|
| `WMG_TRAINER` | no | `local` | `local`, `modal` or `baseten` for `retrain`. |
| `WMG_DEVICE` | no | auto | Force `mps`, `cuda` or `cpu`. |
| `WMG_STUDENT_CFG` | no | `configs/…` | Override the training config YAML. |
| `WMG_BASETEN_GPU` | no | `H100` | GPU type, or `none` for CPU-only. |
| `WMG_BASETEN_SPOT` | no | `1` | `0` for dedicated capacity. |
| `WMG_BASETEN_REMOTE` | no | first in `~/.trussrc` | Which `truss login` remote to use. |
| `BASETEN_TRUSS_AUTH_API_KEY` | **yes** | — | Alternative to `~/.trussrc` (CI only). Use with `BASETEN_TRUSS_AUTH_REMOTE_URL`. |

Set inside the Baseten job by `baseten/config.py` (don't set these yourself): `WMG_MODEL`,
`WMG_TRAIN_ARGS`, `WMG_TAG`, `WMG_FP_*`, `PYTHONUNBUFFERED`. Baseten provides
`BT_PROJECT_CACHE_DIR` and `BT_CHECKPOINT_DIR`.

## Feature analyzer (`feature_analyzer/`)

| Variable | Default | Purpose |
|---|---|---|
| `ANALYZER_DATA` | `feature_analyzer/data` | Uploads and cached analyses. |
| `ANALYZER_RETRAIN_URL` | `http://127.0.0.1:9100` | Retrain service. |
| `ANALYZER_REQUIRE_MODELS` | unset | Tests fail instead of skip when models are missing. |
| `PYTEST_ANALYZER_DATA` | unset | Test data folder. |

## Build-time

| Variable | Purpose |
|---|---|
| `WMG_FEATURES_COMMIT` | Commit stamped into the `wmg_features` Rust crate (`features/build.rs`). |

## Keeping keys out of git

- `.gitignore` (root) ignores `.env`, `.env.*` (except `.env.example`), key files and
  credential files everywhere in the repo.
- Templates (`*.env.example`) contain **names only**, never values.
- Give each key the least access it needs. Baseten training needs only your Baseten API key
  (in `~/.trussrc`); the jobs themselves get no credentials.
- Baseten jobs upload `training/baseten/`, `training/wmg_train/`, `training/configs/` and, the
  first time, the feature caches. Never put a key file in those folders.
- If a key is ever committed, **rotate it first** (ACRCloud console, Fly, Baseten, R2/AWS),
  then remove it from history. Deleting the file in a later commit is not enough.
- Before pushing, `git diff --cached` and look for anything that looks like a key. For an
  automatic check, run [gitleaks](https://github.com/gitleaks/gitleaks) (`gitleaks git .`).
