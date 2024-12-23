use pyo3::prelude::*;

mod ilqr_solver;
mod py_ilqr_solver;

// #[pyfunction]
// fn test(a: usize, f: Bound<'_, PyAny>) -> PyResult<usize> {
//     f.call1((a,))?.extract()
// }

/// A Python module implemented in Rust.
#[pymodule]
fn ilqr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_ilqr_solver::PyILQRSolver>()?;
    Ok(())
}
