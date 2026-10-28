# Live microphone streaming — client guide

This guide shows how to capture microphone audio on a device and stream it to spectro-api.
The server cuts the stream into **30-second clips**. Each clip is saved as a normal upload,
identified with ACRCloud, and analysed. The server also pushes a live mel-spectrogram back
about once a second.

The container cannot reach a microphone, so the device does the capturing and the server
does the processing. Clients send **raw PCM, not compressed audio**. Browser recorders
produce WebM/Opus, which the server cannot decode, and PCM also keeps latency low.

The server side is tracked in [live-streaming-build-plan.md](live-streaming-build-plan.md).

---

## 1. The contract

### Connect

```
ws://<host>:<port>/live?sample_rate=16000[&identify=false]
```

Use `wss://` when the API sits behind TLS. In production that is
`wss://wmg-spectro-api.fly.dev/live?sample_rate=16000`. HTTPS is also what lets browsers use
the microphone on a deployed page.

| Query param | Required | Notes |
|---|---|---|
| `sample_rate` | yes | Integer, 8000–96000. **16000 is recommended**: it is plenty for identification and spectra, and costs about 32 KB/s. |
| `identify` | no | Defaults to `true`. Set it to `false` to skip ACRCloud for this session. |

### Send: binary frames of audio

- **Mono, signed 16-bit, little-endian PCM** at `sample_rate`. There is no header.
- Send a frame every **100–200 ms**: 3,200–6,400 bytes at 16 kHz. Frames do not need to line
  up with samples, because the server carries an odd leftover byte into the next frame.
  Each frame must be ≤ 1 MiB.
- To finish, send the text message `{"type":"stop"}`. The server saves the last partial clip
  if it is at least 3 s long, sends any pending identifications, and then closes the socket.
  Closing the socket yourself also works, but you won't receive the final messages.

### Receive: JSON text messages

Every message has a `type`.

**`session`**: sent once, right after connecting.
```json
{ "type": "session", "session_id": "6f0c…", "clip_seconds": 30, "sample_rate": 16000,
  "spectrum": { "n_mels": 64, "hop_s": 0.032, "db_floor": -80, "freqs": [0, 31.2, …] } }
```

**`spectrum`**: sent about every 1 s, for live display. `frames` is `[frame][mel bin]`, with
bin 0 being the lowest frequency. Values are **absolute dB**, from `db_floor` up to about 0 for
a full-scale signal, so colours are stable from second to second. `t` is the stream time in
seconds of the first frame. If your connection falls behind, the server skips these messages.
```json
{ "type": "spectrum", "t": 12.0, "frames": [[-71.2, -64.0, …], …] }
```

**`clip`**: sent when a 30 s clip has been saved. `id` works with every existing endpoint:
`GET /audio/{id}/stream` for playback, `/audio/{id}/plot/mel` for a PNG, `/data/{kind}`, and
`/stats`.
```json
{ "type": "clip", "index": 0, "id": "b1d2…", "start_s": 0.0, "duration_s": 30.0 }
```

**`identification`**: sent when ACRCloud has finished with a clip. This usually takes 1–3 s
after the matching `clip` message.
```json
{ "type": "identification", "index": 0, "id": "b1d2…", "status": "matched",
  "matches": [{ "source": "music", "title": "…", "artists": ["…"], "album": "…", "score": 100 }] }
```

| `status` | Meaning |
|---|---|
| `matched` | A recording was identified with a score of at least the server's `ACR_MIN_SCORE` (default 50). Use `matches[0]`. |
| `possible_match` | Only low-confidence candidates: humming/cover results, or music matches scoring below the threshold. Show them as "maybe", if at all. |
| `no_match` | Nothing was recognised. |
| `error` | Auth, quota, or network failure. See `message`. |
| `disabled` / `skipped` | Identification is off, either on the server or because you sent `identify=false`. |

**`error`**: a fatal problem, followed by a close with code **1008**.
```json
{ "type": "error", "message": "sample_rate must be between 8000 and 96000" }
```

### Session lifecycle and reconnecting

- One WebSocket is one session. If the connection drops, open a new one, which gets a new
  `session_id`. Clips that were already saved are kept.
- To fetch all clips of a session later: `GET /audio?session_id=<uuid>`. Results are sorted
  by `clip_index`.
- The server limits a session to `LIVE_MAX_MINUTES`, 60 by default.

---

## 2. Capture rules for every platform

1. **Ask for microphone permission** before opening the socket.
2. **Turn off voice processing**: echo cancellation, noise suppression, and automatic gain
   control. These are tuned for speech and remove or distort music, which hurts
   identification.
3. **Mono at 16 kHz.** If the hardware runs at 44.1 or 48 kHz, either resample on the
   device, as the examples below do, or send the native rate in `sample_rate`. The server
   accepts both.
4. **Float to Int16:** `s = round(clamp(x, -1, 1) * 32767)`, written little-endian. All
   current iOS, Android, and desktop CPUs are little-endian, so native byte order is correct.
5. **Background recording.** Mobile OSes suspend apps in the background. If capture must
   continue when the app is not in the foreground, see the notes in each mobile section.

---

## 3. Web browser (JavaScript)

This works in Chrome, Edge, Firefox, and Safari, on desktop and mobile. `getUserMedia` only
works on **HTTPS or `http://localhost`**.

```html
<meta charset="utf-8">
<button id="start">Start</button> <button id="stop" disabled>Stop</button>
<canvas id="spec" width="600" height="128" style="background:#000"></canvas>
<pre id="log"></pre>
<script type="module">
const API = "ws://localhost:9000";      // wss://… in production
const TARGET_RATE = 16000;

// AudioWorklet: averages input samples down to 16 kHz and posts Int16 chunks of ~100 ms.
const workletSrc = `
class PcmSender extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    this.ratio = sampleRate / opts.processorOptions.targetRate; // e.g. 48000/16000 = 3
    this.acc = 0; this.count = 0; this.pos = 0;
    this.size = Math.round(opts.processorOptions.targetRate / 10);   // 100 ms per message
    this.out = new Int16Array(this.size); this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0][0];               // first channel = mono
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i]; this.count++; this.pos += 1;
      if (this.pos >= this.ratio) {        // emit one output sample
        this.pos -= this.ratio;
        const s = Math.max(-1, Math.min(1, this.acc / this.count));
        this.out[this.n++] = Math.round(s * 32767);
        this.acc = 0; this.count = 0;
        if (this.n === this.size) {
          // Transfer the buffer (zero-copy). After this, this.out is detached, so the
          // replacement must use this.size, not this.out.length (which now reads 0).
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Int16Array(this.size); this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-sender", PcmSender);`;

let ws, ctx, stream, node;
const log = (m) => (document.getElementById("log").textContent = m + "\n" + document.getElementById("log").textContent);

document.getElementById("start").onclick = async () => {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  ctx = new AudioContext();               // runs at the device rate (usually 48 kHz)
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([workletSrc], { type: "text/javascript" })));
  node = new AudioWorkletNode(ctx, "pcm-sender", { processorOptions: { targetRate: TARGET_RATE } });
  ctx.createMediaStreamSource(stream).connect(node);  // no need to connect to destination

  ws = new WebSocket(`${API}/live?sample_rate=${TARGET_RATE}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    node.port.onmessage = (e) => ws.readyState === WebSocket.OPEN && ws.send(e.data);
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = (e) => { log(`closed ${e.code} ${e.reason}`); cleanup(); };
  document.getElementById("start").disabled = true;
  document.getElementById("stop").disabled = false;
};

document.getElementById("stop").onclick = () => {
  node?.port && (node.port.onmessage = null);         // stop sending audio
  ws?.send(JSON.stringify({ type: "stop" }));         // server flushes, then closes
};

function cleanup() {
  stream?.getTracks().forEach((t) => t.stop());
  ctx?.close();
  document.getElementById("start").disabled = false;
  document.getElementById("stop").disabled = true;
}

// Live spectrogram: scroll the canvas left and draw each new frame as a 1-px column.
const cv = document.getElementById("spec"), g = cv.getContext("2d");
let floor = -80;
function drawFrames(frames) {
  for (const f of frames) {
    g.drawImage(cv, -1, 0);
    for (let b = 0; b < f.length; b++) {
      const v = Math.max(0, Math.min(1, (f[b] - floor) / -floor));
      g.fillStyle = `hsl(${280 - v * 240}, 90%, ${v * 60}%)`;
      g.fillRect(cv.width - 1, cv.height - (b + 1) * (cv.height / f.length), 1, cv.height / f.length);
    }
  }
}

function handle(msg) {
  switch (msg.type) {
    case "session": floor = msg.spectrum.db_floor; log(`session ${msg.session_id}`); break;
    case "spectrum": drawFrames(msg.frames); break;
    case "clip": log(`clip #${msg.index} saved → ${API.replace("ws", "http")}/audio/${msg.id}/stream`); break;
    case "identification": {
      const m = msg.matches?.[0];
      log(`clip #${msg.index}: ${msg.status}${m ? ` — ${m.title} by ${m.artists.join(", ")}` : ""}`);
      break;
    }
    case "error": log(`error: ${msg.message}`); break;
  }
}
</script>
```

Notes:
- In a React, Vue, or other framework app, keep `ws`, `ctx`, and `node` in a ref or store,
  and call `cleanup()` when the component unmounts.
- Safari only allows an `AudioContext` to start after a user gesture. Starting from a click
  handler, as above, satisfies that.
- To play a saved clip, use `<audio src="http://host/audio/{id}/stream" controls>`.
  Seeking works because the server supports HTTP Range requests.

---

## 4. iOS (Swift)

This uses `AVAudioEngine` for capture, `AVAudioConverter` to convert to 16 kHz Int16 mono, and
`URLSessionWebSocketTask` for transport. It needs iOS 13 or later and no third-party
packages.

**Info.plist**:
```xml
<key>NSMicrophoneUsageDescription</key>
<string>Used to identify and visualise the music around you.</string>
```

```swift
import AVFoundation
import Foundation

final class LiveStreamer: NSObject {
    private let engine = AVAudioEngine()
    private var socket: URLSessionWebSocketTask?
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000,
                                             channels: 1, interleaved: true)!
    var onMessage: (([String: Any]) -> Void)?

    func start(baseURL: String = "ws://localhost:9000") throws {
        // 1. Session: .measurement disables Apple's voice processing (AGC / noise suppression).
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)

        // 2. Socket.
        let url = URL(string: "\(baseURL)/live?sample_rate=16000")!
        socket = URLSession(configuration: .default).webSocketTask(with: url)
        socket?.resume()
        receive()

        // 3. Microphone tap → convert → send.
        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)          // typically 48 kHz Float32
        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw NSError(domain: "LiveStreamer", code: 1)
        }
        // ~100 ms of input per callback.
        input.installTap(onBus: 0, bufferSize: AVAudioFrameCount(inputFormat.sampleRate / 10),
                         format: inputFormat) { [weak self] buffer, _ in
            self?.convertAndSend(buffer, with: converter)
        }
        try engine.start()
    }

    private func convertAndSend(_ buffer: AVAudioPCMBuffer, with converter: AVAudioConverter) {
        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

        var consumed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if consumed { status.pointee = .noDataNow; return nil }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, out.frameLength > 0, let ch = out.int16ChannelData else { return }

        // Int16 interleaved mono, native little-endian byte order.
        let data = Data(bytes: ch[0], count: Int(out.frameLength) * MemoryLayout<Int16>.size)
        socket?.send(.data(data)) { err in if let err { print("send failed: \(err)") } }
    }

    private func receive() {
        socket?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(.string(let text)):
                if let json = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] {
                    DispatchQueue.main.async { self.onMessage?(json) }
                }
                self.receive()
            case .success: self.receive()
            case .failure(let err): print("socket closed: \(err)")
            }
        }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        // Ask the server to flush the last clip; it closes the socket when done.
        socket?.send(.string(#"{"type":"stop"}"#)) { _ in }
    }
}
```

Usage:
```swift
let streamer = LiveStreamer()
streamer.onMessage = { msg in
    switch msg["type"] as? String {
    case "identification":
        let first = (msg["matches"] as? [[String: Any]])?.first
        print(msg["status"] ?? "", first?["title"] ?? "")
    case "spectrum": break   // msg["frames"] as? [[Double]] → draw
    default: print(msg)
    }
}
AVAudioApplication.requestRecordPermission { granted in   // iOS 17+; use AVAudioSession on older
    if granted { try? streamer.start() }
}
```

Notes:
- **Background recording:** in *Signing & Capabilities*, enable *Background Modes → Audio,
  AirPlay, and Picture in Picture*. iOS then keeps the engine running while the app is in the
  background, and shows the orange microphone indicator.
- A real device is required. The Simulator uses the Mac's microphone, which works but does
  not represent device behaviour.
- For local development against a laptop, use the laptop's LAN IP instead of `localhost`. iOS
  also requires an App Transport Security exception for plain `ws://`; use `wss://` in
  production.

---

## 5. Android (Kotlin)

This uses `AudioRecord` for 16-bit PCM capture, an OkHttp WebSocket for transport, and
coroutines for the read loop.

**build.gradle**
```kotlin
implementation("com.squareup.okhttp3:okhttp:4.12.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
```

**AndroidManifest.xml**
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
<!-- Only if recording continues in the background: -->
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE" />
```

```kotlin
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.*
import okhttp3.*
import okio.ByteString.Companion.toByteString
import org.json.JSONObject

class LiveStreamer(
    private val baseUrl: String = "ws://10.0.2.2:9000",   // emulator → host machine
    private val onMessage: (JSONObject) -> Unit,
) {
    private val sampleRate = 16_000
    private val client = OkHttpClient()
    private var socket: WebSocket? = null
    private var recorder: AudioRecord? = null
    private var job: Job? = null

    @SuppressLint("MissingPermission")  // caller must have RECORD_AUDIO granted
    fun start(scope: CoroutineScope) {
        val request = Request.Builder().url("$baseUrl/live?sample_rate=$sampleRate").build()
        socket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) = onMessage(JSONObject(text))
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                stopCapture()
            }
        })

        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val chunkBytes = sampleRate / 10 * 2            // 100 ms of 16-bit mono = 3200 bytes
        // UNPROCESSED skips AGC/noise suppression; not every device supports it.
        val source = MediaRecorder.AudioSource.UNPROCESSED
        recorder = runCatching {
            AudioRecord(source, sampleRate, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, chunkBytes * 4))
                .takeIf { it.state == AudioRecord.STATE_INITIALIZED }
        }.getOrNull() ?: AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, sampleRate,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, chunkBytes * 4))

        recorder!!.startRecording()
        job = scope.launch(Dispatchers.IO) {
            val buf = ByteArray(chunkBytes)
            while (isActive) {
                // ENCODING_PCM_16BIT into a ByteArray yields native (little-endian) samples.
                val n = recorder?.read(buf, 0, buf.size) ?: break
                if (n > 0) socket?.send(buf.toByteString(0, n))
            }
        }
    }

    fun stop() {
        stopCapture()
        socket?.send("""{"type":"stop"}""")   // server flushes the last clip, then closes
    }

    private fun stopCapture() {
        job?.cancel(); job = null
        recorder?.run { stop(); release() }; recorder = null
    }
}
```

Usage, from an Activity or Fragment:
```kotlin
private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
    if (ok) streamer.start(lifecycleScope)
}
private val streamer = LiveStreamer { msg ->
    when (msg.getString("type")) {
        "identification" -> {
            val first = msg.optJSONArray("matches")?.optJSONObject(0)
            Log.i("live", "${msg.getString("status")} ${first?.optString("title") ?: ""}")
        }
        "spectrum" -> { /* msg.getJSONArray("frames") → draw */ }
        else -> Log.i("live", msg.toString())
    }
}
// on button tap:
permission.launch(Manifest.permission.RECORD_AUDIO)
```

Notes:
- **Background recording:** run the streamer inside a foreground `Service` declared with
  `android:foregroundServiceType="microphone"` (required on Android 14 and later), with a
  persistent notification.
- The emulator reaches your computer's `localhost` at `10.0.2.2`. A physical device needs
  the computer's LAN IP. Plain `ws://` also needs
  `android:usesCleartextTraffic="true"` or a network-security config in debug builds.
- OkHttp's `send` queues the data and returns `false` if more than 16 MiB is buffered. At
  32 KB/s that only happens when the network has been stalled for minutes.

---

## 6. Flutter (Dart)

This uses [`record`](https://pub.dev/packages/record) (v5+) to stream PCM and
[`web_socket_channel`](https://pub.dev/packages/web_socket_channel) for transport. It works on
iOS, Android, macOS, and web.

**pubspec.yaml**
```yaml
dependencies:
  record: ^5.1.0
  web_socket_channel: ^3.0.0
```

Platform setup:
- **iOS:** add `NSMicrophoneUsageDescription` to `ios/Runner/Info.plist`, as in §4.
- **Android:** add `RECORD_AUDIO` and `INTERNET` to `AndroidManifest.xml`, as in §5.
- **macOS:** add `com.apple.security.device.audio-input` and
  `com.apple.security.network.client` to both entitlements files.

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class LiveStreamer {
  LiveStreamer({this.baseUrl = 'ws://localhost:9000', required this.onMessage});

  final String baseUrl;
  final void Function(Map<String, dynamic> msg) onMessage;
  static const sampleRate = 16000;

  final _recorder = AudioRecorder();
  WebSocketChannel? _channel;
  StreamSubscription<Uint8List>? _audioSub;

  Future<void> start() async {
    if (!await _recorder.hasPermission()) {
      throw StateError('microphone permission denied');
    }

    _channel = WebSocketChannel.connect(Uri.parse('$baseUrl/live?sample_rate=$sampleRate'));
    await _channel!.ready;
    _channel!.stream.listen(
      (data) => onMessage(jsonDecode(data as String) as Map<String, dynamic>),
      onDone: _stopCapture,
      onError: (_) => _stopCapture(),
    );

    // pcm16bits = signed 16-bit little-endian, which is exactly what the server expects.
    final audio = await _recorder.startStream(const RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: sampleRate,
      numChannels: 1,
      autoGain: false,
      echoCancel: false,
      noiseSuppress: false,
    ));
    _audioSub = audio.listen((chunk) => _channel?.sink.add(chunk));
  }

  Future<void> stop() async {
    await _stopCapture();
    _channel?.sink.add(jsonEncode({'type': 'stop'})); // server flushes, then closes
  }

  Future<void> _stopCapture() async {
    await _audioSub?.cancel();
    _audioSub = null;
    if (await _recorder.isRecording()) await _recorder.stop();
  }

  Future<void> dispose() async {
    await _stopCapture();
    await _channel?.sink.close();
    _recorder.dispose();
  }
}
```

Usage:
```dart
final streamer = LiveStreamer(onMessage: (msg) {
  switch (msg['type']) {
    case 'identification':
      final matches = msg['matches'] as List? ?? const [];
      final title = matches.isNotEmpty ? matches.first['title'] : '';
      debugPrint('${msg['status']} $title');
    case 'spectrum':
      // (msg['frames'] as List) → CustomPainter
      break;
    default:
      debugPrint('$msg');
  }
});
await streamer.start();
// later
await streamer.stop();
```

Notes:
- The chunk size `record` delivers depends on the platform, often 20–100 ms. That is fine
  because the server accepts any frame size.
- On **web**, `record` uses `getUserMedia`, so the HTTPS/localhost rule from §3 applies.
- For background capture on Android, wrap the recording in a foreground service, for
  example with `flutter_foreground_task`. On iOS, enable the *Audio* background mode as in §4.
- Check `record`'s changelog for the exact `RecordConfig` field names in your version. Field
  names have changed between major versions. The ones above are from 5.x.

---

## 7. CLI and testing

These are useful before any app exists. Install
[`websocat`](https://github.com/vi/websocat) (`brew install websocat`) and `ffmpeg`.

**Stream your laptop microphone (macOS):**
```bash
ffmpeg -loglevel error -f avfoundation -i ":0" -ac 1 -ar 16000 -f s16le - \
  | websocat --binary "ws://localhost:9000/live?sample_rate=16000"
```
On Linux, use `-f pulse -i default` or `-f alsa -i default`. To list macOS devices, run
`ffmpeg -f avfoundation -list_devices true -i ""`.

**Stream a file at real-time speed.** This is the easiest way to test identification:
```bash
ffmpeg -loglevel error -re -i song.mp3 -ac 1 -ar 16000 -f s16le - \
  | websocat --binary "ws://localhost:9000/live?sample_rate=16000"
```

`websocat` prints every server message, including the `spectrum` messages. Press Ctrl-C to
stop. With websocat the stream simply ends, so the last partial clip is saved if it is at
least 3 s long.

**Python test client.** This one paces the file in real time, prints a compact summary, and
sends a proper `stop`. It needs `pip install websockets` and ffmpeg.

```python
# live_client.py — usage: python live_client.py song.mp3 [ws://localhost:9000]
import asyncio, json, subprocess, sys, time
import websockets

RATE, CHUNK = 16000, 3200          # 100 ms of 16-bit mono

async def main(path, base="ws://localhost:9000"):
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(RATE),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    async with websockets.connect(f"{base}/live?sample_rate={RATE}", max_size=None) as ws:
        async def sender():
            start = time.monotonic()
            for i in range(0, len(pcm), CHUNK):
                await ws.send(pcm[i:i + CHUNK])
                await asyncio.sleep(max(0, start + (i + CHUNK) / (RATE * 2) - time.monotonic()))
            await ws.send(json.dumps({"type": "stop"}))

        task = asyncio.create_task(sender())
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "spectrum":
                print(f"\rspectrum t={msg['t']:.0f}s", end="", flush=True)
            elif msg["type"] == "identification":
                m = (msg.get("matches") or [{}])[0]
                print(f"\nclip {msg['index']}: {msg['status']} {m.get('title', '')} {m.get('artists', '')}")
            else:
                print("\n" + json.dumps(msg)[:200])
        await task

asyncio.run(main(*sys.argv[1:]))
```

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error` + close 1008 `sample_rate …` | `sample_rate` is missing or outside the 8000–96000 range. |
| `error` `frame too large` | A single binary frame is larger than 1 MiB. Send ~100 ms frames. |
| `error` `session limit reached` | The session ran longer than `LIVE_MAX_MINUTES`. Reconnect to start a new session. |
| Spectrum is all noise or garbled | The byte order or sample format is wrong. The server needs Int16 **little-endian**, not Float32. Also check that `sample_rate` matches what you actually send. |
| Spectrum looks pitch-shifted or stretched | The `sample_rate` you declared doesn't match the real capture rate. |
| No `spectrum` messages | No audio is reaching the socket. Check the microphone permission, and in the browser check that the worklet is connected to the source. |
| Always `no_match` | The level is too low, or voice processing is still on. Disable AEC/NS/AGC, move closer to the source, and confirm the clip with `GET /audio/{clip_id}/stream`. |
| Browser: `getUserMedia` is undefined | The page isn't served over HTTPS or localhost. |
| Mobile can't connect to `localhost` | Use the computer's LAN IP (Android emulator: `10.0.2.2`), and allow cleartext `ws://` in debug builds. |

> **What has been tested:**
> - The browser example was run against the server in Chromium, with a synthetic 440 Hz
>   source in place of a real microphone. That covers capture-to-16 kHz conversion, the
>   socket, spectrum, clips, identification, and Stop.
> - The Python client was run end to end.
> - The ffmpeg + websocat one-liners have not been run.
> - The Swift, Kotlin, and Flutter examples follow current SDK APIs but have not been run on
>   devices. Expect small adjustments for your SDK versions.
