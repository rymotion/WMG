# wmg_frontend

Single-screen Flutter app: tap the mic and it streams raw PCM over a WebSocket to spectro-api's
`/live` endpoint, following `../backend/docs` "Live microphone streaming — client guide". The server
cuts the stream into 30 s clips, identifies each one and pushes a live mel spectrogram back. Tap
again to send `stop`; the server saves the last partial clip (if ≥ 3 s), sends pending
identifications, then closes.

## What the client does (per the guide)

- **Connect** `ws(s)://<API_BASE_URL host:port>/live?sample_rate=16000` (`&identify=false` with
  `--dart-define=LIVE_IDENTIFY=false`). Microphone permission is checked *before* the socket opens.
- **Send** mono s16le PCM at 16 kHz in 3,200-byte (100 ms) binary frames; `{"type":"stop"}` to finish.
- **Capture** with echo cancellation, noise suppression and auto-gain off; Android uses the
  `UNPROCESSED` source, falling back to `VOICE_RECOGNITION` where unsupported.
- **Receive** `session`, `spectrum` (drawn as a scrolling spectrogram), `clip` and
  `identification` (one result card per clip), and `error` (shown; server closes with 1008).
- If the socket drops mid-stream, saved clips stay listed and the next tap opens a new session.
  If the server hasn't closed 20 s after `stop`, the client closes.

## Run

```bash
# backend
cd ../backend && docker compose up --build

# app (defaults to http://localhost:8010, or 10.0.2.2:8010 on the Android emulator)
flutter run -d macos
flutter run -d <device> --dart-define=API_BASE_URL=http://<your-LAN-ip>:8010   # physical phone
```

## Architecture

```
lib/
  core/app_config.dart                  API URL, recording + polling limits
  data/
    models/                             AudioUpload, Identification, TrackMatch (match backend JSON)
    api/spectro_api_client.dart         POST /audio, GET /audio/{id}/identification, GET /audio
    api/live_audio_client.dart          /live WebSocket session: 100 ms PCM frames up, JSON down
    models/live_message.dart            session / spectrum / clip / identification / error
    recorder/audio_recorder_service.dart  `record` PCM stream, voice processing off
    repository/identification_repository.dart  opens live sessions; lists a session's clips
  features/recognizer/
    bloc/                               RecognizerBloc — the event loop
    view/recognizer_page.dart           the one screen
    models/                             Spectrogram (rolling frames), ClipResult
    widgets/                            MicButton (level halo + clip progress), SpectrogramView, ClipResultCard
```

`RecognizerBloc` turns every input into an event and picks a concurrency policy per handler
(`bloc_concurrency`):

| Event | Source | Transformer |
|---|---|---|
| `RecognizerMicPressed` | mic tap, or the 60 min session cap | `droppable` — taps during connect/finish are ignored |
| `RecognizerResetRequested` | app-bar reset (closes the socket) | `sequential` |
| `_RecordingTicked` | 100 ms timer while streaming | concurrent |
| `_PcmChunkSent` | throttled (50 ms) progress; the PCM itself goes straight to the socket | concurrent |
| `_MicStreamFailed` | mic stream error | `sequential` |
| `_ServerMessageReceived` | every JSON message from `/live` | `sequential` |
| `_SessionClosed` | socket closed (after `stop`, after `error`, or a drop) | `sequential` |
| `_FinishTimedOut` | no close 20 s after `stop` | concurrent |

States: `idle → connecting → streaming → finishing → completed` (or `failure`).).
