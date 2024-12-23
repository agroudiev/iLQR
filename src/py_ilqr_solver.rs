use crate::ilqr_solver::ILQRSolver;
use nalgebra::{DMatrix, DVector};
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

    fn solve(&self, x0: Vec<f64>, target: Vec<f64>, dynamics: Bound<'_, PyAny>) -> Vec<Vec<f64>> {
        // Check the input dimensions
        assert!(x0.len() == self.solver.state_dim, "Invalid state dimension");
        assert!(target.len() == self.solver.state_dim, "Invalid target dimension");

        // Convert the input to nalgebra types
        let x0 = DVector::from_row_slice(&x0);
        let target = DVector::from_row_slice(&target);

        // Solve the problem and convert the output
        let us = self.solver.solve(x0, target, |x, u| {
            dynamics.call1((x, u)).unwrap().extract().unwrap()
        });
        us.into_iter().map(|x| x.as_slice().to_vec()).collect()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.solver))
    }
}