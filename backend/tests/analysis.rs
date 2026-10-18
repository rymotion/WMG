mod common;

use common::*;
use spectro_api::{
    acrcloud,
    analysis::{self, MatrixData},
    params::{AnalysisQuery, Kind},
};

fn params() -> spectro_api::params::AnalysisParams {
    AnalysisQuery::default().validate().unwrap()
}

fn argmax(v: impl Iterator<Item = f64>) -> usize {
    v.enumerate().fold((0, f64::MIN), |(bi, bv), (i, x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

#[test]
fn stft_peaks_at_440hz() {
    let y = sine(440.0, 2.0, SR);
    let m = analysis::stft_db(&y, SR, &params()).unwrap();
    let mid = m.values.ncols() / 2;
    let peak = argmax(m.values.column(mid).iter().copied());
    let bin_hz = SR as f64 / 2048.0;
    assert!((m.freqs[peak] - 440.0).abs() <= bin_hz, "peak at {} Hz", m.freqs[peak]);
    assert_eq!(m.values.nrows(), 2048 / 2 + 1);
    // dB is relative to the maximum and floored.
    let max = m.values.iter().cloned().fold(f64::MIN, f64::max);
    let min = m.values.iter().cloned().fold(f64::MAX, f64::min);
    assert!(max.abs() < 1e-9 && min >= -80.0);
}

#[test]
fn chroma_peaks_at_a() {
    let y = sine(440.0, 2.0, SR);
    let m = analysis::chroma(&y, SR, &params()).unwrap();
    assert_eq!(m.values.nrows(), 12);
    let labels = m.labels.as_ref().unwrap();
    let mid = m.values.ncols() / 2;
    let peak = argmax(m.values.column(mid).iter().copied());
    assert_eq!(labels[peak], "A");
}

#[test]
fn mel_and_cqt_shapes() {
    let y = chirp(100.0, 4000.0, 3.0, SR);
    let p = params();
    let mel = analysis::mel_db(&y, SR, &p).unwrap();
    assert_eq!(mel.values.nrows(), 128);
    assert_eq!(mel.freqs.len(), 128);

    let cqt = analysis::cqt_db(&y, SR, &p).unwrap();
    assert_eq!(cqt.values.nrows(), cqt.freqs.len());
    assert!(cqt.values.nrows().is_multiple_of(12));
    // A chirp rises: the loudest CQT bin late in the clip is higher than early on.
    let n = cqt.values.ncols();
    let early = argmax(cqt.values.column(n / 5).iter().copied());
    let late = argmax(cqt.values.column(4 * n / 5).iter().copied());
    assert!(late > early, "early={early} late={late}");
}

#[test]
fn data_downsamples_frames() {
    let y = sine(220.0, 5.0, SR);
    let m = analysis::compute(Kind::Mel, &y, SR, &params()).unwrap();
    let d = MatrixData::from_matrix(&m, 50, 1.5);
    assert!(d.frames <= 50);
    assert_eq!(d.values.len(), d.bins);
    assert_eq!(d.values[0].len(), d.frames);
    assert!(d.times[0] >= 1.5);
}

#[test]
fn stats_on_known_signal() {
    let y = sine(1000.0, 3.0, SR);
    let s = analysis::stats(&y, SR, &params()).unwrap();
    assert!((s.peak - 0.5).abs() < 1e-3);
    assert!((s.rms - 0.5 / 2f64.sqrt()).abs() < 1e-3);
    assert!((s.duration_s - 3.0).abs() < 1e-6);
    // ZCR of a 1 kHz sine is 2000 crossings/sec.
    assert!((s.zero_crossing_rate * SR as f64 - 2000.0).abs() < 20.0);
    assert!((s.spectral_centroid_hz - 1000.0).abs() < 60.0, "centroid {}", s.spectral_centroid_hz);
}

#[test]
fn tempo_of_click_track() {
    // 120 BPM clicks.
    let mut y = vec![0.0; (SR as f64 * 8.0) as usize];
    let (period, n) = ((SR as f64 * 0.5) as usize, y.len());
    for start in (0..n).step_by(period) {
        for (i, v) in y[start..(start + 200).min(n)].iter_mut().enumerate() {
            *v = (1.0 - i as f64 / 200.0) * if i % 2 == 0 { 0.8 } else { -0.8 };
        }
    }
    let bpm = analysis::estimate_tempo(&y, SR, &params()).unwrap().unwrap();
    assert!((bpm - 120.0).abs() < 4.0, "bpm {bpm}");
}

#[test]
fn acr_signature_matches_reference() {
    // Reference computed with Python's hmac/hashlib/base64 per ACRCloud docs.
    assert_eq!(acrcloud::sign("testkey", "testsecret", "1700000000"), "d9SrXTXWNeH1N0Axic0hvCXJlms=");
}

#[test]
fn acr_clip_is_8khz_wav() {
    let y = sine(440.0, 20.0, 44100);
    let wav = acrcloud::clip_to_wav(&y, 44100, 5.0, 10.0).unwrap();
    let r = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
    assert_eq!(r.spec().sample_rate, 8000);
    assert_eq!(r.spec().channels, 1);
    assert_eq!(r.len(), 80_000);
}

#[test]
fn acr_parse_matches() {
    let raw: serde_json::Value = serde_json::from_str(include_str!("fixtures/acr_match.json")).unwrap();
    assert_eq!(acrcloud::status_of(&raw).0, 0);
    let m = acrcloud::parse_matches(&raw);
    assert_eq!(m.len(), 1);
    assert_eq!(m[0].title.as_deref(), Some("Test Song"));
    assert_eq!(m[0].artists, vec!["Test Artist".to_string()]);
    assert_eq!(m[0].album.as_deref(), Some("Test Album"));
    assert_eq!(m[0].play_offset_ms, Some(12000));
    assert_eq!(m[0].source, "music");
}

#[test]
fn acr_parse_humming_with_string_numbers() {
    let raw: serde_json::Value = serde_json::from_str(include_str!("fixtures/acr_humming.json")).unwrap();
    let m = acrcloud::parse_matches(&raw);
    assert_eq!(m.len(), 2);
    assert!(m.iter().all(|x| x.source == "humming"));
    assert_eq!(m[0].duration_ms, Some(194_698));
    assert_eq!(m[0].score, Some(0.61));
    assert!(m[0].external_metadata.is_none());
}
