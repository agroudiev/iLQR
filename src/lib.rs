use pyo3::prelude::*;

mod ilqr_solver;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn test(a: usize, f: Bound<'_, PyAny>) -> PyResult<usize> {
    f.call1((a,))?.extract()
}

/// A Python module implemented in Rust.
#[pymodule]
fn ilqr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(test, m)?)?;
    Ok(())
}
