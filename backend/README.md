# spectro-api

An audio spectrogram analyzer written in Rust and run in Docker. You upload an audio file, and it gives you:

- **Spectrograms** as PNG images or JSON data. These are computed with the [`spectrograms`](https://lib.rs/crates/spectrograms) crate and include linear STFT, mel, chromagram, and CQT, plus a waveform view and summary stats.
- **Playback** through an HTTP stream that supports `Range` requests, so a browser `<audio>` element can play it and seek through it.
- **Track identification** with [ACRCloud](https://docs.acrcloud.com/reference/identification-api/identification-api). This runs automatically in the background after each upload.

The project uses symphonia for decoding (WAV, MP3, FLAC, OGG, AAC/M4A, AIFF), plotters for PNG output, and axum for the HTTP server. All of these are pure Rust, so the container needs no ffmpeg, fontconfig, or Python.

## Run

```bash
cp .env.example .env        # then fill in ACR_ACCESS_KEY / ACR_ACCESS_SECRET
docker compose up --build
curl localhost:9000/health
```

To run it locally without Docker, use `cargo run --release`. It loads `.env` and stores data in `./data`.

To run the tests, use `cargo test`. Inside Docker, use `docker build --target test .`.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check. |
| POST | `/audio[?identify=false]` | Upload a file as multipart field `file`. Responds `201` with the metadata and `identification.status = "pending"`. |
| GET | `/audio` | List uploads, newest first. |
| GET | `/audio/{id}` | Get metadata, including the latest identification result. |
| DELETE | `/audio/{id}` | Delete the file and its cached plots. Responds `204`. |
| GET | `/audio/{id}/stream` | Get the original file. Supports `Range` and returns `206`. Use this for playback. |
| GET | `/audio/{id}/identification` | Get the identification result. Poll this until `status != "pending"`. |
| POST | `/audio/{id}/identify` | Re-run identification. Responds `202`. |
| GET | `/audio/{id}/stats` | Get duration, peak/RMS (dBFS), zero-crossing rate, spectral centroid, and a tempo estimate. |
| GET | `/audio/{id}/plot/{kind}` | Get a PNG. `kind` is one of `waveform`, `stft`, `mel`, `chroma`, or `cqt`. |
| GET | `/audio/{id}/data/{kind}` | Get JSON with `{bins, frames, times[], freqs[] or labels[], values[bin][frame]}`. `kind` is one of `stft`, `mel`, `chroma`, or `cqt`. |

### Analysis query params

`plot`, `data`, and `stats` accept these parameters. Plots and data are cached on disk for each combination of parameters.

| Param | Default | Notes |
|---|---|---|
| `n_fft` | 2048 | Power of 2, from 64 to 16384. |
| `hop` | 512 | At most `n_fft`. |
| `window` | `hanning` | `hanning`, `hamming`, or `blackman`. |
| `n_mels` | 128 | Mel only. |
| `fmin`, `fmax` | per kind | Frequency range in Hz. |
| `db_floor` | -80 | dB values are relative to the maximum, like librosa's `ref=np.max`. |
| `cmap` | `magma` | `magma`, `viridis`, `inferno`, or `gray`. |
| `width`, `height` | 1200×500 | PNG size. |
| `start`, `end` | full file | Time window to analyze, in seconds. |
| `max_frames` | 2000 | Maximum number of frames in a `data` response. |

Invalid parameters return `422`, unsupported or undecodable audio returns `415`, and an unknown id returns `404`.

### Identification statuses

| Status | Meaning |
|---|---|
| `pending` | The background job is queued or running. |
| `matched` | A recording was identified (ACR bucket `music` or `custom_files`) with a score of at least `ACR_MIN_SCORE`. |
| `possible_match` | ACR returned only low-confidence candidates: `humming` results, or music matches scoring below `ACR_MIN_SCORE`. Check `matches[].score`. |
| `no_match` | ACR code `1001` for every window tried. |
| `error` | An auth, quota, or network failure. See `code` and `message`. |
| `disabled` | No credentials are configured. |
| `skipped` | The file was uploaded with `identify=false`. |

The job sends 10-second clips (8 kHz mono WAV) taken at 0 s, 30 s, and 60 s, and stops at the first `matched` result.

### Live microphone streaming

`GET /live?sample_rate=16000` upgrades to a WebSocket.

- **Client sends:** mono Int16 little-endian PCM as binary frames, and `{"type":"stop"}` to finish.
- **Server sends:**
  - `session`, once at the start.
  - `spectrum`, about once a second: absolute-dB mel frames for a live display.
  - `clip`, every 30 s: the clip is saved as a normal upload, so `/audio/{id}/stream` and `/plot` work on it.
  - `identification`, the ACRCloud result for each clip.

`GET /audio?session_id=…` lists the clips from one session.

- Client code for the browser, iOS, Android, Flutter, and the command line: [docs/live-streaming-client.md](docs/live-streaming-client.md)
- Server build checklist: [docs/live-streaming-build-plan.md](docs/live-streaming-build-plan.md)

### Example

```bash
ID=$(curl -s -F file=@song.mp3 localhost:9000/audio | jq -r .id)
curl -s localhost:9000/audio/$ID/identification | jq
curl -o mel.png "localhost:9000/audio/$ID/plot/mel?cmap=viridis&start=10&end=40"
open "http://localhost:9000/audio/$ID/stream"   # plays in the browser
```

## Configuration (env / `.env`)

| Var | Default | |
|---|---|---|
| `PORT` | 8000 (9000 in Docker) | Port the server listens on. |
| `DATA_DIR` | `./data` (`/data` in Docker) | |
| `MAX_UPLOAD_MB` | 100 | |
| `MAX_DURATION_SECS` | 1200 | Files longer than this are rejected. |
| `CORS_ORIGINS` | `*` | A comma-separated list of allowed origins. |
| `ACR_HOST`, `ACR_ACCESS_KEY`, `ACR_ACCESS_SECRET` | – | ACRCloud project credentials. If they are missing, identification is disabled. |
| `ACR_MIN_SCORE` | 50 | Music matches scoring below this are reported as `possible_match`. Steady tones and noise can score around 30. |
| `IDENTIFY_CONCURRENCY` | 2 | The maximum number of ACRCloud requests in flight at once. |
| `LIVE_CLIP_SECS` | 30 | Length of each clip saved from `/live`. |
| `LIVE_MAX_MINUTES` | 60 | Maximum length of one `/live` session. |
| `HOST_PORT` | 9000 | Host port that docker compose publishes. |

`.env` is listed in `.gitignore` and `.dockerignore`. Compose passes it to the container at runtime, so it is never built into an image.

## Layout

- `src/analysis.rs`: wrappers around `spectrograms` and the stats calculations.
- `src/plot.rs` and `src/colormap.rs`: PNG rendering.
- `src/decode.rs`: audio decoding with symphonia.
- `src/acrcloud.rs` and `src/identify.rs`: the signed ACRCloud client and the background jobs.
- `src/storage.rs`: the `AudioStore` trait and its filesystem implementation.
- `src/live.rs` and `src/ingest.rs`: the `/live` WebSocket handler and the save path it shares with uploads.

Postgres/Supabase support is planned in [docs/future-postgres-plan.md](docs/future-postgres-plan.md).
