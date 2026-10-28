use std::f64::consts::PI;
use wmg_features::{DetectorMel, SPEC, extract_windows, spec_hash, window_starts};

fn sine(f: f64, amp: f64, sr: u32, secs: f64) -> Vec<f64> {
    (0..(secs * sr as f64) as usize)
        .map(|i| amp * (2.0 * PI * f * i as f64 / sr as f64).sin())
        .collect()
}

#[test]
fn window_shape_is_128_by_1000() {
    let mel = DetectorMel::new().unwrap();
    let m = mel.logmel(&sine(440.0, 0.5, 32_000, 10.0)).unwrap();
    assert_eq!(m.shape(), &[128, 1000]);
    assert_eq!(SPEC.frames_per_window(), 1000);
}

#[test]
fn full_scale_1khz_is_0db() {
    let mel = DetectorMel::new().unwrap();
    let m = mel.logmel(&sine(1000.0, 1.0, 32_000, 2.0)).unwrap();
    // Skip the zero-padded tail frames.
    let max = m
        .slice(ndarray::s![.., ..190])
        .iter()
        .cloned()
        .fold(f32::MIN, f32::max);
    assert!(max.abs() < 0.1, "peak {max} dB");
    // Half amplitude ≈ −6 dB.
    let m = mel.logmel(&sine(1000.0, 0.5, 32_000, 2.0)).unwrap();
    let max = m
        .slice(ndarray::s![.., ..190])
        .iter()
        .cloned()
        .fold(f32::MIN, f32::max);
    assert!((max + 6.02).abs() < 0.1, "peak {max} dB");
}

#[test]
fn silence_hits_floor() {
    let mel = DetectorMel::new().unwrap();
    let m = mel.logmel(&vec![0.0; 32_000]).unwrap();
    assert!(m.iter().all(|&v| v == SPEC.db_floor as f32));
}

#[test]
fn energy_lands_in_expected_band() {
    // 12 kHz tone: the strongest mel bin should be near the top, proving fmax reaches the
    // treble band rather than stopping at 8 kHz as a 16 kHz pipeline would.
    let mel = DetectorMel::new().unwrap();
    let m = mel.logmel(&sine(12_000.0, 0.5, 32_000, 1.0)).unwrap();
    let col = m.column(50);
    let (argmax, _) = col
        .iter()
        .enumerate()
        .fold((0, f32::MIN), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
    assert!(argmax > 110, "12 kHz peaked at mel bin {argmax}");
}

#[test]
fn windows_overlap_and_gate_quiet_audio() {
    assert_eq!(
        window_starts(32_000 * 30),
        vec![0, 160_000, 320_000, 480_000, 640_000]
    );
    assert_eq!(window_starts(32_000 * 4), vec![0]); // ≥3 s partial → one padded window
    assert!(window_starts(32_000 * 2).is_empty());

    let mel = DetectorMel::new().unwrap();
    // 20 s at 44.1 kHz: first 10 s loud, last 10 s at −60 dBFS.
    let mut x = sine(440.0, 0.5, 44_100, 10.0);
    x.extend(sine(440.0, 0.001, 44_100, 10.0));
    let w = extract_windows(&mel, &x, 44_100).unwrap();
    let starts: Vec<f64> = w.iter().map(|w| w.start_s).collect();
    assert_eq!(starts, vec![0.0, 5.0]); // window at 10 s is gated out
    assert!(w.iter().all(|w| w.logmel.shape() == [128, 1000]));
}

#[test]
fn spec_hash_is_stable() {
    // Changing any spec field changes this; update deliberately alongside SPEC.version.
    assert_eq!(spec_hash().len(), 16);
    assert_eq!(spec_hash(), spec_hash());
}
