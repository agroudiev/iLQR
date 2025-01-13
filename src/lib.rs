use pyo3::prelude::*;

mod ilqr_solver;
mod py_ilqr_errors;
mod py_ilqr_solver;
mod py_utils;

/// A Python module implemented in Rust.
#[pymodule]
fn ilqr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_ilqr_solver::PyILQRSolver>()?;
    m.add_class::<py_ilqr_errors::InstableError>()?;
    Ok(())
}
