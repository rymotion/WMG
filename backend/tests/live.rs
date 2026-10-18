mod common;

use std::{sync::Arc, time::Duration};

use axum::{Router, http::header, routing::post};
use common::*;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use spectro_api::{
    AppState, app,
    config::{AcrConfig, Config},
    storage::FsStore,
};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const RATE: u32 = 16000;

/// Stub ACRCloud that always returns a match.
async fn stub_acr() -> String {
    let stub = Router::new().route(
        "/v1/identify",
        post(|| async { ([(header::CONTENT_TYPE, "application/json")], include_str!("fixtures/acr_match.json")) }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, stub).await.unwrap() });
    format!("http://{addr}")
}

/// Real server on a random port. Returns (host:port, tempdir).
async fn server_with(clip_secs: f64, max_minutes: f64) -> (String, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let cfg = Config {
        port: 0,
        data_dir: dir.path().to_path_buf(),
        max_upload_bytes: 20 * 1024 * 1024,
        max_duration_secs: 600.0,
        cors_origins: vec!["*".into()],
        live_clip_secs: clip_secs,
        live_max_minutes: max_minutes,
        acr: AcrConfig {
            base_url: stub_acr().await,
            access_key: "k".into(),
            access_secret: "s".into(),
            concurrency: 2,
            min_score: 50.0,
        },
    };
    let store = Arc::new(FsStore::new(&cfg.data_dir).await.unwrap());
    let router = app(AppState::new(cfg, store));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    (addr.to_string(), dir)
}

/// Server with 2-second clips.
async fn server(max_minutes: f64) -> (String, tempfile::TempDir) {
    server_with(2.0, max_minutes).await
}

fn pcm(samples: &[f64]) -> Vec<u8> {
    samples.iter().flat_map(|&s| ((s * 32767.0) as i16).to_le_bytes()).collect()
}

/// Read every text message until the server closes; returns (messages, close code).
async fn collect<S>(ws: &mut S) -> (Vec<Value>, Option<u16>)
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let mut msgs = Vec::new();
    let mut code = None;
    while let Ok(Some(Ok(m))) = tokio::time::timeout(Duration::from_secs(30), ws.next()).await {
        match m {
            Message::Text(t) => msgs.push(serde_json::from_str(&t).unwrap()),
            Message::Close(f) => {
                code = f.map(|f| u16::from(f.code));
                break;
            }
            _ => {}
        }
    }
    (msgs, code)
}

#[tokio::test]
async fn streams_clips_spectrum_and_identification() {
    let (addr, _dir) = server(60.0).await;
    let (mut ws, _) = connect_async(format!("ws://{addr}/live?sample_rate={RATE}")).await.unwrap();

    // 5 s of 440 Hz in 100 ms frames; one frame is split at an odd byte boundary.
    let bytes = pcm(&sine(440.0, 5.0, RATE));
    let frame = (RATE / 10 * 2) as usize;
    let mut chunks: Vec<&[u8]> = bytes.chunks(frame).collect();
    let (a, b) = chunks[7].split_at(1601);
    chunks.splice(7..8, [a, b]);
    for c in chunks {
        ws.send(Message::Binary(c.to_vec().into())).await.unwrap();
    }
    ws.send(Message::Text(json!({"type": "stop"}).to_string().into())).await.unwrap();

    let (msgs, code) = collect(&mut ws).await;
    assert_eq!(code, Some(1000), "{msgs:?}");
    let of = |t: &str| msgs.iter().filter(|m| m["type"] == t).cloned().collect::<Vec<_>>();

    let session = &of("session")[0];
    assert_eq!(session["clip_seconds"], 2.0);
    assert_eq!(session["spectrum"]["n_mels"], 64);
    let session_id = session["session_id"].as_str().unwrap().to_string();

    let spectra = of("spectrum");
    assert!(spectra.len() >= 4, "only {} spectrum messages", spectra.len());
    assert_eq!(spectra[0]["frames"][0].as_array().unwrap().len(), 64);
    // Times increase and the 440 Hz tone is near 0 dB (calibrated to a full-scale sine ≈ 0 dB; amplitude 0.5 ≈ -6 dB).
    assert!(spectra[1]["t"].as_f64().unwrap() > spectra[0]["t"].as_f64().unwrap());
    let peak = spectra[1]["frames"][3].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).fold(f64::MIN, f64::max);
    assert!((-12.0..=1.0).contains(&peak), "peak {peak} dB");

    // 5 s with 2 s clips → clips 0 and 1; the 1 s remainder is below the 3 s minimum.
    let mut clips = of("clip");
    clips.sort_by_key(|c| c["index"].as_u64());
    assert_eq!(clips.len(), 2, "{clips:?}");
    assert_eq!(clips[1]["start_s"], 2.0);
    assert!((clips[0]["duration_s"].as_f64().unwrap() - 2.0).abs() < 1e-9);

    let idents = of("identification");
    assert_eq!(idents.len(), 2);
    assert!(idents.iter().all(|i| i["status"] == "matched" && i["matches"][0]["title"] == "Test Song"));
    assert!(idents.iter().all(|i| i.get("raw").is_none()));

    // Clips are ordinary uploads, grouped by session.
    let list: Value = reqwest::get(format!("http://{addr}/audio?session_id={session_id}")).await.unwrap().json().await.unwrap();
    let list = list.as_array().unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0]["clip_index"], 0);
    assert_eq!(list[1]["clip_index"], 1);
    assert_eq!(list[0]["identification"]["status"], "matched");
    let id = clips[0]["id"].as_str().unwrap();
    let png = reqwest::get(format!("http://{addr}/audio/{id}/plot/mel")).await.unwrap();
    assert_eq!(png.status(), 200);
    assert_eq!(&png.bytes().await.unwrap()[..4], b"\x89PNG");
}

#[tokio::test]
async fn final_partial_clip_is_saved_when_long_enough() {
    // 4 s clips; 7.5 s of audio → clip 0 (4 s) + a 3.5 s remainder, which meets the 3 s minimum.
    let (addr, _dir) = server_with(4.0, 60.0).await;
    let (mut ws, _) = connect_async(format!("ws://{addr}/live?sample_rate={RATE}&identify=false")).await.unwrap();
    ws.send(Message::Binary(pcm(&sine(220.0, 7.5, RATE)).into())).await.unwrap();
    ws.send(Message::Text(r#"{"type":"stop"}"#.into())).await.unwrap();
    let (msgs, code) = collect(&mut ws).await;
    assert_eq!(code, Some(1000));
    let mut clips: Vec<_> = msgs.iter().filter(|m| m["type"] == "clip").collect();
    clips.sort_by_key(|c| c["index"].as_u64());
    assert_eq!(clips.len(), 2);
    assert_eq!(clips[1]["start_s"], 4.0);
    assert!((clips[1]["duration_s"].as_f64().unwrap() - 3.5).abs() < 1e-9);
    // identify=false: no identification messages.
    assert_eq!(msgs.iter().filter(|m| m["type"] == "identification").count(), 0);
}

#[tokio::test]
async fn rejects_bad_sample_rate() {
    let (addr, _dir) = server(60.0).await;
    let (mut ws, _) = connect_async(format!("ws://{addr}/live?sample_rate=44")).await.unwrap();
    let (msgs, code) = collect(&mut ws).await;
    assert_eq!(code, Some(1008));
    assert_eq!(msgs[0]["type"], "error");
    assert!(msgs[0]["message"].as_str().unwrap().contains("sample_rate"));
}

#[tokio::test]
async fn rejects_oversized_frame() {
    let (addr, _dir) = server(60.0).await;
    let (mut ws, _) = connect_async(format!("ws://{addr}/live?sample_rate={RATE}")).await.unwrap();
    ws.send(Message::Binary(vec![0u8; 2 * 1024 * 1024].into())).await.unwrap();
    let (msgs, code) = collect(&mut ws).await;
    assert_eq!(code, Some(1008));
    assert!(msgs.iter().any(|m| m["type"] == "error" && m["message"].as_str().unwrap().contains("too large")));
}

#[tokio::test]
async fn enforces_session_limit() {
    // 1 minute cap: send 61 s in big frames.
    let (addr, _dir) = server(1.0).await;
    let (mut ws, _) = connect_async(format!("ws://{addr}/live?sample_rate=8000&identify=false")).await.unwrap();
    let second = pcm(&vec![0.0; 8000]);
    for _ in 0..61 {
        if ws.send(Message::Binary(second.clone().into())).await.is_err() {
            break;
        }
    }
    let (msgs, code) = collect(&mut ws).await;
    assert_eq!(code, Some(1008));
    assert!(msgs.iter().any(|m| m["message"] == "session limit reached"));
}
