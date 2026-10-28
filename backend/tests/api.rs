mod common;

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode, header},
    routing::post,
};
use common::*;
use http_body_util::BodyExt;
use serde_json::Value;
use spectro_api::{
    AppState, app,
    config::{AcrConfig, Config},
    storage::FsStore,
};
use tower::ServiceExt;

const BOUNDARY: &str = "XTESTBOUNDARYX";

fn config(dir: &std::path::Path, acr_url: &str) -> Config {
    Config {
        port: 0,
        data_dir: dir.to_path_buf(),
        max_upload_bytes: 20 * 1024 * 1024,
        max_duration_secs: 600.0,
        cors_origins: vec!["*".into()],
        live_clip_secs: 30.0,
        live_max_minutes: 60.0,
        acr: AcrConfig {
            base_url: acr_url.to_string(),
            access_key: if acr_url.is_empty() { String::new() } else { "k".into() },
            access_secret: if acr_url.is_empty() { String::new() } else { "s".into() },
            concurrency: 2,
            min_score: 50.0,
        },
    }
}

async fn router(dir: &std::path::Path, acr_url: &str) -> Router {
    let cfg = config(dir, acr_url);
    let store = Arc::new(FsStore::new(&cfg.data_dir).await.unwrap());
    app(AppState::new(cfg, store))
}

fn multipart(filename: &str, bytes: &[u8]) -> Body {
    let mut body = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    Body::from(body)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let headers = res.headers().clone();
    let body = res.into_body().collect().await.unwrap().to_bytes().to_vec();
    (status, headers, body)
}

async fn upload(app: &Router, filename: &str, bytes: &[u8], query: &str) -> (StatusCode, Value) {
    let req = Request::post(format!("/audio{query}"))
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={BOUNDARY}"))
        .body(multipart(filename, bytes))
        .unwrap();
    let (status, _, body) = send(app, req).await;
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

#[tokio::test]
async fn upload_analyse_stream_delete() {
    let dir = tempfile::tempdir().unwrap();
    let app = router(dir.path(), "").await;
    let wav = wav_bytes(&chirp(200.0, 3000.0, 4.0, SR), SR);

    let (status, meta) = upload(&app, "chirp.wav", &wav, "").await;
    assert_eq!(status, StatusCode::CREATED, "{meta}");
    assert_eq!(meta["sample_rate"], SR);
    assert!((meta["duration_s"].as_f64().unwrap() - 4.0).abs() < 0.01);
    // No credentials: identification is reported as disabled, not pending.
    assert_eq!(meta["identification"]["status"], "disabled");
    let id = meta["id"].as_str().unwrap().to_string();

    let (status, _, body) = send(&app, get("/audio")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap().as_array().unwrap().len(), 1);

    for kind in ["waveform", "stft", "mel", "chroma", "cqt"] {
        let (status, headers, body) = send(&app, get(&format!("/audio/{id}/plot/{kind}?width=600&height=300"))).await;
        assert_eq!(status, StatusCode::OK, "{kind}: {}", String::from_utf8_lossy(&body));
        assert_eq!(headers[header::CONTENT_TYPE], "image/png");
        assert_eq!(&body[..8], b"\x89PNG\r\n\x1a\n");
        std::fs::write(dir.path().join(format!("{kind}.png")), &body).unwrap();
    }
    // Second request is served from cache with the same ETag.
    let (_, h1, _) = send(&app, get(&format!("/audio/{id}/plot/mel?width=600&height=300"))).await;
    assert!(h1.contains_key(header::ETAG));

    let (status, _, body) = send(&app, get(&format!("/audio/{id}/data/mel?max_frames=100&start=1&end=3"))).await;
    assert_eq!(status, StatusCode::OK);
    let data: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(data["bins"], 128);
    assert!(data["frames"].as_u64().unwrap() <= 100);
    assert!(data["times"][0].as_f64().unwrap() >= 1.0);

    let (status, _, body) = send(&app, get(&format!("/audio/{id}/stats"))).await;
    assert_eq!(status, StatusCode::OK);
    let stats: Value = serde_json::from_slice(&body).unwrap();
    assert!((stats["peak"].as_f64().unwrap() - 0.5).abs() < 0.01);

    // Playback: full file and a byte range.
    let (status, headers, body) = send(&app, get(&format!("/audio/{id}/stream"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.len(), wav.len());
    assert_eq!(headers[header::ACCEPT_RANGES], "bytes");
    assert!(headers[header::CONTENT_TYPE].to_str().unwrap().contains("wav"));
    let req = Request::get(format!("/audio/{id}/stream")).header(header::RANGE, "bytes=0-1023").body(Body::empty()).unwrap();
    let (status, headers, body) = send(&app, req).await;
    assert_eq!(status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(body.len(), 1024);
    assert!(headers[header::CONTENT_RANGE].to_str().unwrap().starts_with("bytes 0-1023/"));

    let req = Request::delete(format!("/audio/{id}")).body(Body::empty()).unwrap();
    assert_eq!(send(&app, req).await.0, StatusCode::NO_CONTENT);
    assert_eq!(send(&app, get(&format!("/audio/{id}"))).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn rejects_bad_input() {
    let dir = tempfile::tempdir().unwrap();
    let app = router(dir.path(), "").await;

    let (status, _) = upload(&app, "notes.txt", b"hello", "").await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    let (status, _) = upload(&app, "fake.wav", b"definitely not audio", "").await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    // The failed upload left nothing behind.
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);

    let missing = uuid::Uuid::new_v4();
    assert_eq!(send(&app, get(&format!("/audio/{missing}"))).await.0, StatusCode::NOT_FOUND);
    assert_eq!(send(&app, get(&format!("/audio/{missing}/plot/mel"))).await.0, StatusCode::NOT_FOUND);

    let wav = wav_bytes(&sine(440.0, 1.0, SR), SR);
    let (_, meta) = upload(&app, "tone.wav", &wav, "").await;
    let id = meta["id"].as_str().unwrap();
    assert_eq!(send(&app, get(&format!("/audio/{id}/plot/mel?n_fft=1000"))).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(send(&app, get(&format!("/audio/{id}/plot/mel?start=5&end=2"))).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(send(&app, get(&format!("/audio/{id}/plot/bogus"))).await.0, StatusCode::BAD_REQUEST);
}

/// Spawn a stub ACRCloud server answering with `fixture` and return its base URL.
async fn stub_acr(fixture: &'static str) -> String {
    let stub = Router::new().route(
        "/v1/identify",
        post(move |body: axum::body::Bytes| async move {
            // The real API receives multipart with the signed fields.
            let text = String::from_utf8_lossy(&body);
            assert!(text.contains("name=\"signature\"") && text.contains("name=\"sample\""));
            ([(header::CONTENT_TYPE, "application/json")], fixture)
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, stub).await.unwrap() });
    format!("http://{addr}")
}

async fn wait_for_identification(app: &Router, id: &str) -> Value {
    for _ in 0..100 {
        let (_, _, body) = send(app, get(&format!("/audio/{id}/identification"))).await;
        let v: Value = serde_json::from_slice(&body).unwrap();
        if v["status"] != "pending" {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("identification never finished");
}

#[tokio::test]
async fn identification_runs_in_background() {
    let dir = tempfile::tempdir().unwrap();
    let url = stub_acr(include_str!("fixtures/acr_match.json")).await;
    let app = router(dir.path(), &url).await;

    let wav = wav_bytes(&sine(440.0, 12.0, SR), SR);
    let (status, meta) = upload(&app, "song.wav", &wav, "").await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(meta["identification"]["status"], "pending");
    let id = meta["id"].as_str().unwrap();

    let ident = wait_for_identification(&app, id).await;
    assert_eq!(ident["status"], "matched", "{ident}");
    assert_eq!(ident["matches"][0]["title"], "Test Song");
    assert_eq!(ident["matches"][0]["artists"][0], "Test Artist");

    // Embedded in the resource too.
    let (_, _, body) = send(&app, get(&format!("/audio/{id}"))).await;
    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()["identification"]["status"], "matched");

    // identify=false skips the job.
    let (_, meta) = upload(&app, "song2.wav", &wav, "?identify=false").await;
    assert_eq!(meta["identification"]["status"], "skipped");
}

#[tokio::test]
async fn identification_no_match() {
    let dir = tempfile::tempdir().unwrap();
    let url = stub_acr(include_str!("fixtures/acr_nomatch.json")).await;
    let app = router(dir.path(), &url).await;
    let wav = wav_bytes(&sine(440.0, 70.0, 8000), 8000);
    let (_, meta) = upload(&app, "tone.wav", &wav, "").await;
    let ident = wait_for_identification(&app, meta["id"].as_str().unwrap()).await;
    assert_eq!(ident["status"], "no_match");
    assert_eq!(ident["code"], 1001);
    // All three windows (0s, 30s, 60s) were tried; the last one is reported.
    assert_eq!(ident["window_offset_s"], 60.0);
}

#[tokio::test]
async fn identification_humming_only_is_possible_match() {
    let dir = tempfile::tempdir().unwrap();
    let url = stub_acr(include_str!("fixtures/acr_humming.json")).await;
    let app = router(dir.path(), &url).await;
    let wav = wav_bytes(&sine(440.0, 5.0, SR), SR);
    let (_, meta) = upload(&app, "hum.wav", &wav, "").await;
    let ident = wait_for_identification(&app, meta["id"].as_str().unwrap()).await;
    assert_eq!(ident["status"], "possible_match");
    assert_eq!(ident["matches"][0]["source"], "humming");
}

#[tokio::test]
async fn low_score_music_is_possible_match() {
    let dir = tempfile::tempdir().unwrap();
    let url = stub_acr(include_str!("fixtures/acr_lowscore.json")).await;
    let app = router(dir.path(), &url).await;
    let wav = wav_bytes(&sine(440.0, 5.0, SR), SR);
    let (_, meta) = upload(&app, "tone.wav", &wav, "").await;
    let ident = wait_for_identification(&app, meta["id"].as_str().unwrap()).await;
    assert_eq!(ident["status"], "possible_match");
    assert_eq!(ident["matches"][0]["source"], "music");
    assert_eq!(ident["matches"][0]["score"], 28.0);
}

#[tokio::test]
async fn identification_transport_error() {
    let dir = tempfile::tempdir().unwrap();
    // Nothing listens here.
    let app = router(dir.path(), "http://127.0.0.1:9").await;
    let wav = wav_bytes(&sine(440.0, 5.0, SR), SR);
    let (_, meta) = upload(&app, "tone.wav", &wav, "").await;
    let ident = wait_for_identification(&app, meta["id"].as_str().unwrap()).await;
    assert_eq!(ident["status"], "error");
}

#[tokio::test]
async fn personalised_head_upload_list_and_fetch() {
    let dir = tempfile::tempdir().unwrap();
    let state = AppState::new(config(dir.path(), "http://127.0.0.1:9"), Arc::new(FsStore::new(dir.path()).await.unwrap()));
    let app = app(state);
    let body = serde_json::json!({
        "format": "wmg-personal-head/1", "id": "20261023120000-abc", "base_model_run": "20261019-080656",
        "feature_spec_hash": "9f612611d1fccf8a", "device": "test", "created_at": "2026-10-23T12:00:00Z",
        "head": {"weights": [0.1, -0.2, 0.3], "bias": -0.5},
        "train": {"ai_windows": 6, "human_windows": 6}, "samples": [{"label": 1}, {"label": 0}],
    });
    let req = |method: &str, uri: &str, body: Option<&Value>| {
        Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json")
            .body(body.map_or(Body::empty(), |b| Body::from(b.to_string()))).unwrap()
    };
    let res = app.clone().oneshot(req("POST", "/models/personalized", Some(&body))).await.unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let created: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(created["samples"], 2);
    let id = created["id"].as_str().unwrap().to_string();
    assert!(id.contains("20261023120000-abc"));

    let res = app.clone().oneshot(req("GET", "/models/personalized", None)).await.unwrap();
    let list: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(list.as_array().unwrap().len(), 1);

    let res = app.clone().oneshot(req("GET", &format!("/models/personalized/{id}"), None)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let back: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(back["head"]["bias"], -0.5);

    // Wrong format and path traversal are rejected.
    let bad = serde_json::json!({"format": "other"});
    let res = app.clone().oneshot(req("POST", "/models/personalized", Some(&bad))).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let res = app.clone().oneshot(req("GET", "/models/personalized/..%2Fsecret", None)).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}
