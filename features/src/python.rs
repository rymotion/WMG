//! `wmg_features` Python module (built with maturin, `--features python`).

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{DetectorMel, FeatureError, SPEC};

impl From<FeatureError> for PyErr {
    fn from(e: FeatureError) -> Self {
        PyValueError::new_err(e.to_string())
    }
}

fn to_vec(samples: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    samples.as_array().to_vec()
}

/// `(start_s, rms_dbfs, logmel)` as returned to Python.
type PyWindow<'py> = (f64, f64, Bound<'py, PyArray2<f32>>);

#[pyclass(name = "DetectorMel", frozen)]
struct PyDetectorMel(DetectorMel);

#[pymethods]
impl PyDetectorMel {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self(DetectorMel::new()?))
    }

    /// Log-mel `[n_mels, ceil(len / hop)]` (float32 dB) for 32 kHz mono float64 samples.
    fn logmel<'py>(
        &self,
        py: Python<'py>,
        samples: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = to_vec(&samples);
        let m = py.detach(|| self.0.logmel(&x))?;
        Ok(m.into_pyarray(py))
    }

    /// Full pipeline for mono float64 `samples` at `sr` Hz.
    /// Returns a list of `(start_s, rms_dbfs, logmel[n_mels, frames])`.
    fn extract_windows<'py>(
        &self,
        py: Python<'py>,
        samples: PyReadonlyArray1<'py, f64>,
        sr: u32,
    ) -> PyResult<Vec<PyWindow<'py>>> {
        let x = to_vec(&samples);
        let windows = py.detach(|| crate::extract_windows(&self.0, &x, sr))?;
        Ok(windows
            .into_iter()
            .map(|w| (w.start_s, w.rms_dbfs, w.logmel.into_pyarray(py)))
            .collect())
    }

    /// As `extract_windows`, with each window's standardised side features as a 4th item:
    /// `(start_s, rms_dbfs, logmel, side[N_SIDE])`.
    fn extract_windows_side<'py>(
        &self,
        py: Python<'py>,
        samples: PyReadonlyArray1<'py, f64>,
        sr: u32,
    ) -> PyResult<Vec<(f64, f64, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>)>> {
        let x = to_vec(&samples);
        let windows = py.detach(|| crate::extract_windows(&self.0, &x, sr))?;
        Ok(windows
            .into_iter()
            .map(|w| (w.start_s, w.rms_dbfs, w.logmel.into_pyarray(py), w.side.to_vec().into_pyarray(py)))
            .collect())
    }
}

/// Raw (unstandardised) side features of one 32 kHz mono window; NaN where undefined.
#[pyfunction]
fn side_features_raw<'py>(py: Python<'py>, samples: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let x = to_vec(&samples);
    py.detach(|| crate::side_features_raw(&x)).to_vec().into_pyarray(py)
}

/// Standardised side features of one 32 kHz mono window (what the device appends).
#[pyfunction]
fn side_features<'py>(py: Python<'py>, samples: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f32>> {
    let x = to_vec(&samples);
    py.detach(|| crate::side_features(&x)).to_vec().into_pyarray(py)
}

/// Resample mono float64 `samples` from `sr_in` to `sr_out` Hz.
#[pyfunction]
fn resample<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'py, f64>,
    sr_in: u32,
    sr_out: u32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if sr_in == 0 || sr_out == 0 {
        return Err(FeatureError::BadSampleRate.into());
    }
    let x = to_vec(&samples);
    Ok(py
        .detach(|| crate::resample(&x, sr_in, sr_out))
        .into_pyarray(py))
}

#[pyfunction]
fn rms_dbfs(samples: PyReadonlyArray1<'_, f64>) -> f64 {
    crate::rms_dbfs(samples.as_array().as_slice().unwrap_or(&to_vec(&samples)))
}

#[pymodule]
fn wmg_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDetectorMel>()?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(rms_dbfs, m)?)?;
    m.add_function(wrap_pyfunction!(side_features_raw, m)?)?;
    m.add_function(wrap_pyfunction!(side_features, m)?)?;
    m.add("SIDE_FEATURES", crate::SIDE_FEATURES.to_vec())?;
    m.add("SIDE_SCALING", crate::SIDE_SCALING.to_vec())?;
    m.add("N_SIDE", crate::N_SIDE)?;
    m.add("SPEC_JSON", crate::spec_json())?;
    m.add("SPEC_HASH", crate::spec_hash())?;
    m.add("COMMIT", crate::COMMIT)?;
    m.add("SAMPLE_RATE", SPEC.sample_rate)?;
    m.add("N_MELS", SPEC.n_mels)?;
    m.add("FRAMES_PER_WINDOW", SPEC.frames_per_window())?;
    Ok(())
}
