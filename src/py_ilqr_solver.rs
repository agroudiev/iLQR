use crate::ilqr_solver::ILQRSolver;
use nalgebra::DMatrix;
use pyo3::{pyclass, pymethods};
use pyo3::prelude::*;

#[pyclass(name = "ILQRSolver")]
#[derive(Debug)]
/// A Python wrapper for `ILQRSolver`
pub struct PyILQRSolver {
    solver: ILQRSolver,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyILQRSolver {
    #[new]
    fn new(
        state_dim: usize,
        control_dim: usize,
        // dynamics: Bound<'_, PyAny>,
        Q: Vec<Vec<f64>>,
        R: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            solver: ILQRSolver::new(
                state_dim,
                control_dim,
                DMatrix::from_row_slice(Q.len(), Q[0].len(), &Q.concat()),
                DMatrix::from_row_slice(R.len(), R[0].len(), &R.concat()),
            ),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.solver))
    }
}