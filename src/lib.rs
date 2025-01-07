use pyo3::prelude::*;

mod ilqr_solver;
mod py_ilqr_solver;

/// A Python module implemented in Rust.
#[pymodule]
fn ilqr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_ilqr_solver::PyILQRSolver>()?;
    Ok(())
}
