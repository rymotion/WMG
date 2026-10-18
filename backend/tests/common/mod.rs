#![allow(dead_code)]

use std::{f64::consts::PI, io::Cursor};

pub const SR: u32 = 22050;

pub fn sine(freq: f64, secs: f64, sr: u32) -> Vec<f64> {
    (0..(secs * sr as f64) as usize).map(|i| 0.5 * (2.0 * PI * freq * i as f64 / sr as f64).sin()).collect()
}

/// Linear chirp from `f0` to `f1` Hz.
pub fn chirp(f0: f64, f1: f64, secs: f64, sr: u32) -> Vec<f64> {
    let k = (f1 - f0) / secs;
    (0..(secs * sr as f64) as usize)
        .map(|i| {
            let t = i as f64 / sr as f64;
            0.5 * (2.0 * PI * (f0 * t + 0.5 * k * t * t)).sin()
        })
        .collect()
}

pub fn wav_bytes(samples: &[f64], sr: u32) -> Vec<u8> {
    let spec = hound::WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut c = Cursor::new(Vec::new());
    {
        let mut w = hound::WavWriter::new(&mut c, spec).unwrap();
        for &s in samples {
            w.write_sample((s * i16::MAX as f64) as i16).unwrap();
        }
        w.finalize().unwrap();
    }
    c.into_inner()
}
