//! Profile every song under a directory: native-rate spectrum (encoder cutoff, long-term
//! average spectrum), levels, stereo width, and detector log-mel statistics computed with the
//! exact `wmg-features` pipeline. Writes one JSON object per song (JSONL).
//!
//!   cargo run --release --example profile_dir -- <root> <out.jsonl>
//!
//! Files whose path relative to `<root>` starts with `Music/` are treated as duplicates of the
//! same path without that prefix and skipped when the original exists.

use realfft::RealFftPlanner;
use serde_json::json;
use std::{
    collections::HashSet,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Mutex, atomic::{AtomicUsize, Ordering}},
};
use symphonia::core::{
    audio::SampleBuffer, codecs::{CODEC_TYPE_NULL, DecoderOptions}, errors::Error as SymError,
    formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};
use wmg_features::{DetectorMel, SPEC, extract_windows};

const AUDIO_EXT: &[&str] = &["m4a", "mp3", "wav", "aif", "aiff", "flac", "ogg", "mp4", "aac"];
/// FFT size for the native-rate long-term average spectrum.
const LTAS_FFT: usize = 8192;
/// Detector band edges in Hz (bass / mid / treble), matching the planned BandNet split.
const BANDS: [(f64, f64); 3] = [(20.0, 250.0), (250.0, 4000.0), (4000.0, 16_000.0)];

struct Decoded {
    left: Vec<f32>,
    right: Vec<f32>,
    sr: u32,
    channels: usize,
    codec: String,
}

fn decode(path: &Path) -> Result<Decoded, String> {
    let mss = MediaSourceStream::new(Box::new(File::open(path).map_err(|e| e.to_string())?), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| e.to_string())?
        .format;
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).ok_or("no audio track")?;
    let track_id = track.id;
    let codec = symphonia::default::get_codecs()
        .get_codec(track.codec_params.codec)
        .map(|d| d.short_name.to_string())
        .unwrap_or_else(|| "unknown".into());
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default()).map_err(|e| e.to_string())?;

    let (mut left, mut right) = (Vec::new(), Vec::new());
    let (mut sr, mut channels) = (0, 0);
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e.to_string()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e.to_string()),
        };
        let spec = *decoded.spec();
        sr = spec.rate;
        channels = spec.channels.count().max(1);
        let needed = decoded.capacity() as u64;
        if buf.as_ref().is_none_or(|b| (b.capacity() as u64) < needed * channels as u64) {
            buf = Some(SampleBuffer::<f32>::new(needed, spec));
        }
        let b = buf.as_mut().expect("initialised");
        b.copy_interleaved_ref(decoded);
        for frame in b.samples().chunks_exact(channels) {
            left.push(frame[0]);
            right.push(*frame.get(1).unwrap_or(&frame[0]));
        }
    }
    if left.is_empty() || sr == 0 {
        return Err("no decodable audio".into());
    }
    Ok(Decoded { left, right, sr, channels, codec })
}

fn db(p: f64) -> f64 {
    10.0 * p.max(1e-30).log10()
}

/// Long-term average power spectrum of `x` (Hann, non-overlapping frames), in dB per bin.
fn ltas(x: &[f64]) -> Vec<f64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(LTAS_FFT);
    let win: Vec<f64> =
        (0..LTAS_FFT).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / LTAS_FFT as f64).cos()).collect();
    let mut acc = vec![0.0; LTAS_FFT / 2 + 1];
    let (mut inp, mut out) = (fft.make_input_vec(), fft.make_output_vec());
    let mut n = 0;
    for chunk in x.chunks_exact(LTAS_FFT) {
        for ((d, s), w) in inp.iter_mut().zip(chunk).zip(&win) {
            *d = s * w;
        }
        fft.process(&mut inp, &mut out).expect("fft sizes match");
        for (a, c) in acc.iter_mut().zip(&out) {
            *a += c.norm_sqr();
        }
        n += 1;
    }
    acc.iter().map(|&p| db(p / n.max(1) as f64)).collect()
}

/// Highest frequency whose LTAS (smoothed over ~200 Hz) is within 70 dB of the 200 Hz–4 kHz
/// median — i.e. where the encoder's low-pass (or the recording's bandwidth) ends.
fn cutoff_hz(spec_db: &[f64], sr: u32) -> f64 {
    let hz = sr as f64 / LTAS_FFT as f64;
    let half = ((100.0 / hz).ceil() as usize).max(1);
    let smooth: Vec<f64> = (0..spec_db.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(spec_db.len()));
            spec_db[a..b].iter().sum::<f64>() / (b - a) as f64
        })
        .collect();
    let mut ref_bins: Vec<f64> = smooth[(200.0 / hz) as usize..(4000.0 / hz) as usize].to_vec();
    ref_bins.sort_by(f64::total_cmp);
    let thresh = ref_bins[ref_bins.len() / 2] - 70.0;
    smooth.iter().rposition(|&v| v > thresh).map_or(0.0, |i| i as f64 * hz)
}

/// LTAS resampled onto 96 log-spaced points from 20 Hz to Nyquist (for plotting/comparison).
fn ltas_log(spec_db: &[f64], sr: u32) -> (Vec<f64>, Vec<f64>) {
    let hz = sr as f64 / LTAS_FFT as f64;
    let nyq = sr as f64 / 2.0;
    let edges: Vec<f64> = (0..=96).map(|i| 20.0 * (nyq / 20.0).powf(i as f64 / 96.0)).collect();
    let mut freqs = Vec::new();
    let mut vals = Vec::new();
    for w in edges.windows(2) {
        let (a, b) = ((w[0] / hz) as usize, ((w[1] / hz).ceil() as usize).min(spec_db.len() - 1).max((w[0] / hz) as usize + 1));
        let p = spec_db[a..b].iter().map(|d| 10f64.powf(d / 10.0)).sum::<f64>() / (b - a) as f64;
        freqs.push((w[0] * w[1]).sqrt().round());
        vals.push((db(p) * 10.0).round() / 10.0);
    }
    (freqs, vals)
}

fn mel_band_ranges() -> Vec<(usize, usize)> {
    // Slaney mel centres for the detector spec, to map Hz bands onto mel bin indices.
    let mel = |f: f64| if f < 1000.0 { 3.0 * f / 200.0 } else { 15.0 + 27.0 * (f / 1000.0).ln() / 6.4f64.ln() };
    let (lo, hi) = (mel(SPEC.fmin), mel(SPEC.fmax));
    let centre_mel = |i: usize| lo + (hi - lo) * (i + 1) as f64 / (SPEC.n_mels + 1) as f64;
    BANDS
        .iter()
        .map(|&(a, b)| {
            let bins: Vec<usize> = (0..SPEC.n_mels).filter(|&i| (mel(a)..mel(b)).contains(&centre_mel(i))).collect();
            (*bins.first().unwrap_or(&0), *bins.last().unwrap_or(&0) + 1)
        })
        .collect()
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn profile(path: &Path, root: &Path, mel: &DetectorMel, bands: &[(usize, usize)]) -> serde_json::Value {
    let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy().to_string();
    let parts: Vec<&str> = rel.split('/').collect();
    let base = json!({ "path": rel, "artist": parts.first(), "album": parts.get(parts.len().saturating_sub(2)) });
    let d = match decode(path) {
        Ok(d) => d,
        Err(e) => return json!({ "path": rel, "error": e }),
    };
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let duration = d.left.len() as f64 / d.sr as f64;

    let mono: Vec<f64> = d.left.iter().zip(&d.right).map(|(&l, &r)| (l as f64 + r as f64) / 2.0).collect();
    let side_e: f64 = d.left.iter().zip(&d.right).map(|(&l, &r)| ((l - r) as f64 / 2.0).powi(2)).sum();
    let mid_e: f64 = mono.iter().map(|v| v * v).sum();
    let peak = d.left.iter().chain(&d.right).fold(0f32, |m, v| m.max(v.abs())) as f64;
    let rms_db = db(mid_e / mono.len() as f64);
    let clip_frac = d.left.iter().chain(&d.right).filter(|v| v.abs() >= 0.999).count() as f64 / (2 * d.left.len()) as f64;

    let spec = ltas(&mono);
    let cutoff = cutoff_hz(&spec, d.sr);
    let (ltas_f, ltas_db) = ltas_log(&spec, d.sr);

    let windows = match extract_windows(mel, &mono, d.sr) {
        Ok(w) => w,
        Err(e) => return json!({ "path": rel, "error": e.to_string() }),
    };
    let n_mels = SPEC.n_mels;
    let mut sum = vec![0.0f64; n_mels];
    let mut sumsq = vec![0.0f64; n_mels];
    let mut frames = 0usize;
    let mut treble_flatness = Vec::new();
    let mut flux = Vec::new();
    for w in &windows {
        let m = &w.logmel;
        for (j, col) in m.columns().into_iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                sum[i] += v as f64;
                sumsq[i] += (v as f64).powi(2);
            }
            frames += 1;
            // Spectral flatness (geometric / arithmetic mean of power) over treble bins.
            let (a, b) = bands[2];
            let p: Vec<f64> = col.iter().skip(a).take(b - a).map(|&v| 10f64.powf(v as f64 / 10.0)).collect();
            let am = p.iter().sum::<f64>() / p.len() as f64;
            let gm = (p.iter().map(|v| v.max(1e-30).ln()).sum::<f64>() / p.len() as f64).exp();
            treble_flatness.push(gm / am.max(1e-30));
            if j > 0 {
                let prev = m.column(j - 1);
                flux.push(col.iter().zip(prev.iter()).map(|(c, p)| (c - p).max(0.0) as f64).sum::<f64>() / n_mels as f64);
            }
        }
    }
    let f = frames.max(1) as f64;
    let mean: Vec<f64> = sum.iter().map(|s| s / f).collect();
    let std: Vec<f64> = sumsq.iter().zip(&mean).map(|(s, m)| (s / f - m * m).max(0.0).sqrt()).collect();
    let band_mean: Vec<f64> = bands.iter().map(|&(a, b)| round2(mean[a..b].iter().sum::<f64>() / (b - a) as f64)).collect();
    let avg = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };

    let mut out = base;
    let o = out.as_object_mut().expect("object");
    for (k, v) in [
        ("file_ext", json!(path.extension().and_then(|e| e.to_str()).map(str::to_lowercase))),
        ("codec", json!(d.codec)),
        ("sample_rate", json!(d.sr)),
        ("channels", json!(d.channels)),
        ("duration_s", json!(round2(duration))),
        ("bitrate_kbps", json!(round2(size as f64 * 8.0 / duration / 1000.0))),
        ("peak_dbfs", json!(round2(20.0 * peak.max(1e-9).log10()))),
        ("rms_dbfs", json!(round2(rms_db))),
        ("crest_db", json!(round2(20.0 * peak.max(1e-9).log10() - rms_db))),
        ("clipped_frac", json!(clip_frac)),
        ("side_to_mid_db", json!(round2(db(side_e) - db(mid_e)))),
        ("cutoff_hz", json!(cutoff.round())),
        ("n_windows", json!(windows.len())),
        ("n_windows_possible", json!(wmg_features::window_starts((duration * SPEC.sample_rate as f64) as usize).len())),
        ("band_mean_db", json!({"bass": band_mean[0], "mid": band_mean[1], "treble": band_mean[2]})),
        ("treble_flatness", json!(round2(avg(&treble_flatness) * 100.0) / 100.0)),
        ("spectral_flux", json!(round2(avg(&flux)))),
        ("mel_mean_db", json!(mean.iter().map(|&v| round2(v)).collect::<Vec<_>>())),
        ("mel_std_db", json!(std.iter().map(|&v| round2(v)).collect::<Vec<_>>())),
        ("ltas_hz", json!(ltas_f)),
        ("ltas_db", json!(ltas_db)),
        ("feature_spec_hash", json!(wmg_features::spec_hash())),
        ("feature_crate_commit", json!(wmg_features::COMMIT)),
    ] {
        o.insert(k.into(), v);
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| AUDIO_EXT.contains(&e.to_lowercase().as_str())) {
            out.push(p);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (root, out_path) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let mut files = Vec::new();
    walk(&root, &mut files);
    files.sort();

    // Drop `Music/<x>` when `<x>` also exists.
    let rels: HashSet<PathBuf> = files.iter().filter_map(|p| p.strip_prefix(&root).ok().map(Path::to_path_buf)).collect();
    let (mut keep, mut dups) = (Vec::new(), Vec::new());
    for p in files {
        let rel = p.strip_prefix(&root).expect("under root").to_path_buf();
        match rel.strip_prefix("Music") {
            Ok(orig) if rels.contains(orig) => dups.push(rel),
            _ => keep.push(p),
        }
    }
    eprintln!("{} audio files to profile, {} duplicates under Music/ skipped", keep.len(), dups.len());

    let mel = DetectorMel::new().expect("detector mel");
    let bands = mel_band_ranges();
    eprintln!("mel band bins (bass, mid, treble): {bands:?}");
    let writer = Mutex::new(BufWriter::new(File::create(&out_path).expect("create output")));
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(p) = keep.get(i) else { break };
                let row = profile(p, &root, &mel, &bands);
                let mut w = writer.lock().expect("writer");
                writeln!(w, "{row}").expect("write");
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 25 == 0 || n == keep.len() {
                    eprintln!("{n}/{}", keep.len());
                }
            });
        }
    });
    writer.into_inner().expect("writer").flush().expect("flush");
    let dup_path = out_path.with_file_name("duplicates.txt");
    std::fs::write(&dup_path, dups.iter().map(|p| p.to_string_lossy() + "\n").collect::<String>()).expect("write dups");
}
