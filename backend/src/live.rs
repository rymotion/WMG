//! `GET /live`: microphone streaming over WebSocket.
//!
//! Protocol (see `docs/live-streaming-client.md`): the client sends mono Int16 LE PCM as
//! binary frames; the server pushes `session`, `spectrum` (~1 s), `clip` (every
//! `LIVE_CLIP_SECS`) and `identification` JSON messages, and `error` before closing with 1008.

use std::{io::Cursor, sync::Arc, time::Duration};

use axum::{
    extract::{
        Query, State,
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade, close_code},
    },
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{sync::mpsc, task::JoinSet};
use uuid::Uuid;

use crate::{
    AppState,
    analysis::LiveMel,
    ingest::{self, SessionClip},
    storage::IdentificationStatus,
};

const MAX_FRAME_BYTES: usize = 1024 * 1024;
const MIN_FINAL_CLIP_S: f64 = 3.0;
const N_MELS: usize = 64;
const DB_FLOOR: f64 = -80.0;
/// How long to wait for in-flight clip saves/identifications after the client stops.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(25);

#[derive(Debug, Deserialize)]
pub struct LiveQuery {
    sample_rate: Option<String>,
    identify: Option<String>,
}

struct Session {
    id: Uuid,
    sample_rate: u32,
    identify: bool,
}

fn origin_allowed(state: &AppState, headers: &HeaderMap) -> bool {
    let origins = &state.config.cors_origins;
    if origins.iter().any(|o| o == "*") {
        return true;
    }
    // Non-browser clients send no Origin; browsers always do.
    match headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        None => true,
        Some(o) => origins.iter().any(|allowed| allowed == o),
    }
}

fn parse_query(q: &LiveQuery) -> Result<Session, String> {
    let sample_rate = q
        .sample_rate
        .as_deref()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|sr| (8000..=96000).contains(sr))
        .ok_or("sample_rate must be between 8000 and 96000")?;
    let identify = match q.identify.as_deref() {
        None | Some("true" | "1" | "yes") => true,
        Some("false" | "0" | "no") => false,
        Some(_) => return Err("identify must be true or false".into()),
    };
    Ok(Session { id: Uuid::new_v4(), sample_rate, identify })
}

pub async fn handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(q): Query<LiveQuery>,
    headers: HeaderMap,
) -> Response {
    if !origin_allowed(&state, &headers) {
        return (StatusCode::FORBIDDEN, "origin not allowed").into_response();
    }
    // Allow larger messages at the protocol layer so oversized frames get a readable error.
    ws.max_message_size(MAX_FRAME_BYTES * 4)
        .on_upgrade(move |socket| run(socket, state, parse_query(&q)))
}

/// Outgoing messages go through a channel to a single writer task.
#[derive(Clone)]
struct Out(mpsc::Sender<Message>);

impl Out {
    async fn json(&self, v: Value) {
        let _ = self.0.send(Message::Text(v.to_string().into())).await;
    }

    /// Lossy send for high-rate messages: dropped when the client is behind.
    fn try_json(&self, v: Value) {
        let _ = self.0.try_send(Message::Text(v.to_string().into()));
    }

    async fn fatal(&self, msg: &str) {
        self.json(json!({ "type": "error", "message": msg })).await;
        self.close(close_code::POLICY, msg).await;
    }

    async fn close(&self, code: u16, reason: &str) {
        let frame = CloseFrame { code, reason: reason.to_string().into() };
        let _ = self.0.send(Message::Close(Some(frame))).await;
    }
}

async fn run(socket: WebSocket, state: AppState, session: Result<Session, String>) {
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<Message>(64);
    let writer = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let closing = matches!(msg, Message::Close(_));
            if sink.send(msg).await.is_err() || closing {
                break;
            }
        }
        let _ = sink.close().await;
    });
    let out = Out(tx);

    let session = match session {
        Ok(s) => s,
        Err(e) => {
            out.fatal(&e).await;
            drop(out);
            let _ = writer.await;
            return;
        }
    };
    let sr = session.sample_rate;
    let mel = match LiveMel::new(sr, N_MELS, DB_FLOOR) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            out.fatal(&e.to_string()).await;
            drop(out);
            let _ = writer.await;
            return;
        }
    };

    tracing::info!(session = %session.id, sample_rate = sr, "live session started");
    out.json(json!({
        "type": "session",
        "session_id": session.id,
        "clip_seconds": state.config.live_clip_secs,
        "sample_rate": sr,
        "spectrum": {
            "n_mels": mel.n_mels,
            "hop_s": mel.hop as f64 / sr as f64,
            "db_floor": mel.floor_db,
            "freqs": mel.freqs.iter().map(|f| (f * 10.0).round() / 10.0).collect::<Vec<_>>(),
        },
    }))
    .await;

    let clip_len = (state.config.live_clip_secs * sr as f64) as usize;
    let max_samples = (state.config.live_max_minutes * 60.0 * sr as f64) as u64;
    let mut clip_buf: Vec<i16> = Vec::with_capacity(clip_len);
    let mut spec_buf: Vec<f64> = Vec::new();
    let mut spec_consumed: u64 = 0; // samples already turned into spectrum frames
    let mut total: u64 = 0;
    let mut carry: Option<u8> = None;
    let mut clip_index: u32 = 0;
    let mut clip_start: u64 = 0;
    let mut jobs = JoinSet::new();
    let mut fatal: Option<&'static str> = None;

    while let Some(msg) = stream.next().await {
        let bytes = match msg {
            Ok(Message::Binary(b)) => b,
            Ok(Message::Text(t)) => {
                if serde_json::from_str::<Value>(&t).ok().and_then(|v| v.get("type").cloned()) == Some(json!("stop")) {
                    break;
                }
                continue;
            }
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(_) => continue,
        };
        if bytes.len() > MAX_FRAME_BYTES {
            fatal = Some("frame too large (max 1 MiB)");
            break;
        }

        // Int16 LE → samples, carrying an odd trailing byte into the next frame.
        let mut new_samples: Vec<i16> = Vec::with_capacity(bytes.len() / 2 + 1);
        let mut data: &[u8] = &bytes;
        if let Some(lo) = carry.take() {
            match data.split_first() {
                Some((&hi, rest)) => {
                    new_samples.push(i16::from_le_bytes([lo, hi]));
                    data = rest;
                }
                None => carry = Some(lo),
            }
        }
        new_samples.extend(data.as_chunks::<2>().0.iter().map(|&c| i16::from_le_bytes(c)));
        if data.len() % 2 == 1 {
            carry = data.last().copied();
        }

        total += new_samples.len() as u64;
        spec_buf.extend(new_samples.iter().map(|&s| s as f64 / 32768.0));
        clip_buf.extend_from_slice(&new_samples);

        // Live spectrum, roughly once per second of audio.
        if spec_buf.len() >= sr as usize + mel.n_fft {
            let buf = std::mem::take(&mut spec_buf);
            let m = Arc::clone(&mel);
            if let Ok((frames, used, buf)) =
                tokio::task::spawn_blocking(move || m.frames(&buf).map(|(f, u)| (f, u, buf))).await.unwrap_or_else(|e| Err(e.into()))
            {
                if !frames.is_empty() {
                    let rounded: Vec<Vec<f32>> =
                        frames.iter().map(|f| f.iter().map(|&v| ((v * 10.0).round() / 10.0) as f32).collect()).collect();
                    out.try_json(json!({ "type": "spectrum", "t": spec_consumed as f64 / sr as f64, "frames": rounded }));
                }
                spec_consumed += used as u64;
                spec_buf = buf[used..].to_vec();
            }
        }

        // Completed clips.
        while clip_buf.len() >= clip_len {
            let rest = clip_buf.split_off(clip_len);
            let clip = std::mem::replace(&mut clip_buf, rest);
            spawn_clip(&mut jobs, &state, &session, &out, clip_index, clip_start, clip);
            clip_index += 1;
            clip_start += clip_len as u64;
        }

        if total > max_samples {
            fatal = Some("session limit reached");
            break;
        }
    }

    // Flush the final partial clip.
    if clip_buf.len() as f64 >= MIN_FINAL_CLIP_S * sr as f64 {
        spawn_clip(&mut jobs, &state, &session, &out, clip_index, clip_start, std::mem::take(&mut clip_buf));
    }
    if tokio::time::timeout(DRAIN_TIMEOUT, async { while jobs.join_next().await.is_some() {} }).await.is_err() {
        tracing::warn!(session = %session.id, "live session closed with jobs still running");
        jobs.abort_all();
    }

    match fatal {
        Some(msg) => out.fatal(msg).await,
        None => out.close(close_code::NORMAL, "done").await,
    }
    drop(out);
    let _ = writer.await;
    tracing::info!(session = %session.id, clips = clip_index, seconds = total as f64 / sr as f64, "live session ended");
}

/// Save one clip as an upload, announce it, then identify it and announce the result.
fn spawn_clip(
    jobs: &mut JoinSet<()>,
    state: &AppState,
    session: &Session,
    out: &Out,
    index: u32,
    start_sample: u64,
    samples: Vec<i16>,
) {
    let (state, out) = (state.clone(), out.clone());
    let (session_id, sr, identify) = (session.id, session.sample_rate, session.identify);
    jobs.spawn(async move {
        let duration_s = samples.len() as f64 / sr as f64;
        let wav = match tokio::task::spawn_blocking(move || encode_wav(&samples, sr)).await {
            Ok(Ok(w)) => w,
            _ => {
                out.json(json!({ "type": "error", "message": format!("failed to encode clip {index}") })).await;
                return;
            }
        };
        let filename = format!("live-{session_id}-{index:03}.wav");
        let clip = SessionClip { session_id, index };
        let meta = match ingest::ingest(&state, filename, "wav".into(), &wav, identify, Some(clip)).await {
            Ok(m) => m,
            Err(e) => {
                out.json(json!({ "type": "error", "message": format!("failed to save clip {index}: {e}") })).await;
                return;
            }
        };
        out.json(json!({
            "type": "clip",
            "index": index,
            "id": meta.id,
            "start_s": start_sample as f64 / sr as f64,
            "duration_s": duration_s,
        }))
        .await;

        if !identify {
            return;
        }
        let ident = if meta.identification.status == IdentificationStatus::Pending {
            state.identifier.run_and_store(meta.id).await
        } else {
            meta.identification // e.g. `disabled`
        };
        let mut msg = serde_json::to_value(&ident).unwrap_or_else(|_| json!({}));
        if let Value::Object(map) = &mut msg {
            map.remove("raw");
            map.insert("type".into(), json!("identification"));
            map.insert("index".into(), json!(index));
            map.insert("id".into(), json!(meta.id));
        }
        out.json(msg).await;
    });
}

fn encode_wav(samples: &[i16], sr: u32) -> Result<Vec<u8>, hound::Error> {
    let spec = hound::WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut cursor = Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    {
        let mut w = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in samples {
            w.write_sample(s)?;
        }
        w.finalize()?;
    }
    Ok(cursor.into_inner())
}
