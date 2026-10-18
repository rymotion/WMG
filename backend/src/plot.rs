//! PNG rendering with plotters (bitmap backend, embedded font: no system deps).

use std::{io::Cursor, sync::Once};

use plotters::{coord::Shift, prelude::*};

use crate::{
    analysis::Matrix,
    colormap::Lut,
    error::{AppError, AppResult},
    params::{AnalysisParams, Kind},
};

const FONT: &str = "sans-serif";
static FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");
static INIT: Once = Once::new();

fn init_fonts() {
    INIT.call_once(|| {
        for style in [FontStyle::Normal, FontStyle::Bold, FontStyle::Italic, FontStyle::Oblique] {
            // Only fails on a malformed font file, which is embedded at compile time.
            let _ = plotters::style::register_font(FONT, style, FONT_BYTES);
        }
    });
}

fn plot_err(e: impl std::fmt::Display) -> AppError {
    AppError::Internal(format!("plotting failed: {e}"))
}

fn encode_png(buf: Vec<u8>, w: u32, h: u32) -> AppResult<Vec<u8>> {
    let img = image::RgbImage::from_raw(w, h, buf).ok_or_else(|| plot_err("bad buffer size"))?;
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).map_err(plot_err)?;
    Ok(out.into_inner())
}

fn fmt_hz(f: f64) -> String {
    if f >= 1000.0 { format!("{:.1}k", f / 1000.0) } else { format!("{f:.0}") }
}

fn y_desc(kind: Kind) -> &'static str {
    match kind {
        Kind::Stft => "Frequency (Hz)",
        Kind::Mel => "Frequency (Hz, mel-spaced)",
        Kind::Cqt => "Frequency (Hz, log-spaced)",
        Kind::Chroma => "Pitch class",
        Kind::Waveform => "Amplitude",
    }
}

/// Spectrogram-style heatmap with a colour bar.
pub fn heatmap_png(m: &Matrix, p: &AnalysisParams, title: &str, offset_s: f64) -> AppResult<Vec<u8>> {
    init_fonts();
    let (w, h) = (p.width, p.height);
    let mut buf = vec![255u8; (w * h * 3) as usize];
    let lut = Lut::new(p.cmap);
    let n_bins = m.values.nrows();
    let n_frames = m.values.ncols();
    let t0 = offset_s;
    let t1 = offset_s + m.times.last().copied().unwrap_or(0.0).max(1e-3);

    {
        let root = BitMapBackend::with_buffer(&mut buf, (w, h)).into_drawing_area();
        root.fill(&WHITE).map_err(plot_err)?;
        let (main, bar) = root.split_horizontally(w.saturating_sub(100));

        let mut chart = ChartBuilder::on(&main)
            .caption(title, (FONT, 18))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(72)
            .build_cartesian_2d(t0..t1, -0.5..(n_bins as f64 - 0.5))
            .map_err(plot_err)?;

        let label_for = |y: &f64| -> String {
            let i = y.round().clamp(0.0, (n_bins - 1) as f64) as usize;
            match (&m.labels, m.freqs.get(i)) {
                (Some(l), _) => l.get(i).cloned().unwrap_or_default(),
                (None, Some(&f)) => fmt_hz(f),
                _ => String::new(),
            }
        };
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc("Time (s)")
            .y_desc(y_desc(m.kind))
            .y_labels(if m.labels.is_some() { n_bins } else { 10 })
            .y_label_formatter(&label_for)
            .x_label_formatter(&|x| format!("{x:.1}"))
            .label_style((FONT, 13))
            .axis_desc_style((FONT, 14))
            .draw()
            .map_err(plot_err)?;

        // Fill the plotting area pixel by pixel (nearest-neighbour sampling).
        let area = chart.plotting_area().strip_coord_spec();
        let (aw, ah) = area.dim_in_pixel();
        for px in 0..aw {
            let col = ((px as f64 + 0.5) / aw as f64 * n_frames as f64) as usize;
            let col = col.min(n_frames - 1);
            for py in 0..ah {
                let row = ((1.0 - (py as f64 + 0.5) / ah as f64) * n_bins as f64) as usize;
                let row = row.min(n_bins - 1);
                let [r, g, b] = lut.map(m.values[[row, col]], m.vmin, m.vmax);
                area.draw_pixel((px as i32, py as i32), &RGBColor(r, g, b)).map_err(plot_err)?;
            }
        }

        colorbar(&bar, &lut, m.vmin, m.vmax, m.unit)?;
        root.present().map_err(plot_err)?;
    }
    encode_png(buf, w, h)
}

fn colorbar(area: &DrawingArea<BitMapBackend<'_>, Shift>, lut: &Lut, vmin: f64, vmax: f64, unit: &str) -> AppResult<()> {
    let mut chart = ChartBuilder::on(area)
        .margin_top(38)
        .margin_bottom(50)
        .margin_right(4)
        .right_y_label_area_size(56)
        .build_cartesian_2d(0.0..1.0, vmin..vmax)
        .map_err(plot_err)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .disable_x_axis()
        .y_labels(6)
        .y_desc(unit)
        .label_style((FONT, 12))
        .axis_desc_style((FONT, 12))
        .y_label_formatter(&|v| if vmax - vmin <= 2.0 { format!("{:.1}", v + 0.0) } else { format!("{:.0}", v + 0.0) })
        .draw()
        .map_err(plot_err)?;
    let bar = chart.plotting_area().strip_coord_spec();
    let (bw, bh) = bar.dim_in_pixel();
    for py in 0..bh {
        let v = vmax - (py as f64 + 0.5) / bh as f64 * (vmax - vmin);
        let [r, g, b] = lut.map(v, vmin, vmax);
        for px in 0..bw.min(22) {
            bar.draw_pixel((px as i32, py as i32), &RGBColor(r, g, b)).map_err(plot_err)?;
        }
    }
    Ok(())
}

/// Waveform as a per-pixel min/max envelope.
pub fn waveform_png(samples: &[f64], sr: u32, p: &AnalysisParams, title: &str, offset_s: f64) -> AppResult<Vec<u8>> {
    init_fonts();
    let (w, h) = (p.width, p.height);
    let mut buf = vec![255u8; (w * h * 3) as usize];
    let duration = (samples.len() as f64 / sr as f64).max(1e-3);
    let peak = samples.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let ylim = if peak > 0.0 { (peak * 1.05).min(1.05) } else { 1.0 };

    {
        let root = BitMapBackend::with_buffer(&mut buf, (w, h)).into_drawing_area();
        root.fill(&WHITE).map_err(plot_err)?;
        let mut chart = ChartBuilder::on(&root)
            .caption(title, (FONT, 18))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(offset_s..offset_s + duration, -ylim..ylim)
            .map_err(plot_err)?;
        chart
            .configure_mesh()
            .light_line_style(RGBColor(235, 235, 235))
            .x_desc("Time (s)")
            .y_desc(y_desc(Kind::Waveform))
            .x_label_formatter(&|x| format!("{x:.1}"))
            .y_label_formatter(&|y| format!("{:.2}", if y.abs() < 5e-3 { 0.0 } else { *y }))
            .label_style((FONT, 13))
            .axis_desc_style((FONT, 14))
            .draw()
            .map_err(plot_err)?;

        let cols = chart.plotting_area().dim_in_pixel().0.max(1) as usize;
        let color = RGBColor(52, 101, 164);
        let per_col = samples.len() as f64 / cols as f64;
        chart
            .draw_series((0..cols).filter_map(|i| {
                let a = (i as f64 * per_col) as usize;
                let b = (((i + 1) as f64 * per_col) as usize).clamp(a + 1, samples.len());
                let chunk = samples.get(a..b)?;
                let (lo, hi) = chunk.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
                let t = offset_s + (i as f64 + 0.5) / cols as f64 * duration;
                Some(PathElement::new(vec![(t, lo), (t, hi)], color.stroke_width(1)))
            }))
            .map_err(plot_err)?;
        root.present().map_err(plot_err)?;
    }
    encode_png(buf, w, h)
}
