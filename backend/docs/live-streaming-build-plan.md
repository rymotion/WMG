# Live streaming — server build plan

This is the server-side work behind `GET /live`. The client contract lives in
[live-streaming-client.md](live-streaming-client.md); **that document is the source of truth for
the protocol**. If the two disagree, update both.

## Goal
Clients stream mono Int16 little-endian PCM over a WebSocket. The server does three things:
1. Pushes absolute-dB mel frames back about once per second.
2. Every `LIVE_CLIP_SECS` (30 s), saves the buffered audio as a normal upload, so playback,
   plots, data, and stats work unchanged.
3. Runs ACRCloud identification on each clip and pushes the result back on the same socket.

## Checklist

### 1. Plumbing
- [x] Enable axum's `ws` feature. Add `futures-util`, and `tokio-tungstenite` as a
      dev-dependency.
- [x] `config.rs`: add `LIVE_CLIP_SECS` (default 30) and `LIVE_MAX_MINUTES` (default 60).
- [x] `lib.rs`: add the `/live` route.

### 2. Shared ingest path
- [x] New `src/ingest.rs` with
      `ingest(state, filename, ext, bytes, identify, session: Option<(Uuid, u32)>) -> AudioMeta`.
      It writes the file, decodes it to validate, builds and creates the `AudioMeta`, and
      records the initial identification status.
- [x] `routes/audio.rs::upload` becomes a thin wrapper that keeps the current behaviour,
      including spawning identification.
- [x] `AudioMeta` gains `session_id: Option<Uuid>` and `clip_index: Option<u32>`, both
      `#[serde(default)]` so existing `meta.json` files still load.
- [x] `GET /audio?session_id=<uuid>` filters by session and sorts by `clip_index`.

### 3. Identification hook
- [x] `Identifier::run_and_store(id) -> Identification` is public. It runs the job, stores the
      result, and returns it. `spawn` uses it, and live sessions await it so they can push the
      result.

### 4. Live spectrum
- [x] `analysis::mel_db_absolute(samples, sr, n_fft, hop, n_mels, floor)` uses
      `centre = false` and `10·log10(power)` floored at `floor`. dB is not relative to the
      maximum, so successive chunks share one colour scale.
- [x] Settings: n_fft = 1024 (or 512 below 16 kHz), hop = n_fft/2, and 64 mels.
- [x] Continuity: keep the last `n_fft − hop` samples from each block and prepend them to the
      next one, so frames tile seamlessly. `t` is the start time of each message's first frame.

### 5. WebSocket handler (`src/live.rs`)
- [x] Validate the query: `sample_rate` must be 8000–96000. Otherwise send an `error` and
      close with 1008.
- [x] Check `Origin` against `CORS_ORIGINS` unless it is `*`, because browsers don't apply
      CORS to WebSockets.
- [x] Split the socket. A writer task drains a bounded `mpsc` channel.
      - `spectrum` messages use `try_send` and are dropped when the client lags.
      - `session`, `clip`, `identification`, and `error` messages use `send`.
- [x] Reader:
      - Binary frames longer than 1 MiB produce an `error`.
      - Carry an odd leftover byte into the next frame.
      - Convert Int16 to f64 and append to the clip buffer and the spectrum buffer.
- [x] Every `sample_rate` samples, compute the spectrum in `spawn_blocking` and push it.
- [x] Every `clip_secs × sample_rate` samples:
      - Encode the clip as WAV with `hound` and call `ingest` with
        `filename = live-<session>-<n>.wav` and `identify = false`.
      - Push a `clip` message.
      - If `identify` is set, spawn `run_and_store`, which pushes an `identification`
        message.
- [x] Session cap: past `LIVE_MAX_MINUTES`, send `error` `session limit reached` and close.
- [x] On `{"type":"stop"}` or a close from the client:
      - Save the remainder as a final clip if it is at least 3 s.
      - Wait up to 20 s for pending identifications.
      - Close normally with 1000.

### 6. Docs & tests
- [x] README: add a short "Live microphone streaming" section linking both docs.
- [x] `tests/live.rs`: run the real server on a random port with `LIVE_CLIP_SECS=2` and a stub
      ACRCloud server.
      - Stream 5 s of 16 kHz sine in 100 ms frames, including one odd-length frame, then stop.
      - Expect one `session` message, at least 4 `spectrum` messages with 64 bins, clips 0
        and 1, and no third clip (the 1 s remainder is under 3 s). Expect an
        `identification` for each clip.
      - Expect `GET /audio?session_id=` to return 2 clips, and a working `plot/mel` for a clip.
      - Error cases: a bad `sample_rate`, and an oversized frame.

### 7. Verify
- [x] `cargo test`, `cargo clippy --all-targets`, and `docker build --target test .`
- [x] `docker compose up -d --build` (port 9000)
- [x] Python client from the client guide, run end to end against the container.
- [ ] ffmpeg + websocat one-liners. Not run yet, because the local ffmpeg is broken
      (missing `libbluray`) and websocat isn't installed.
- [x] Browser example in the in-app browser, using a synthetic 440 Hz source. The browser
      pane blocks microphone access.
- [ ] Browser example with a real microphone, in Chrome/Safari on `localhost`.
- [ ] Swift, Kotlin, and Flutter examples on real devices.
