use crate::ilqr_solver::ILQRSolver;
use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

#[pyclass(name = "ILQRSolver")]
#[derive(Debug)]
/// A Python wrapper for `ILQRSolver`
pub struct PyILQRSolver {
    solver: ILQRSolver,
}

// TODO: replace the `Vec` by numpy arrays

#[pymethods]
#[allow(non_snake_case)]
impl PyILQRSolver {
    #[new]
    fn new(state_dim: usize, control_dim: usize, Q: Vec<Vec<f64>>, R: Vec<Vec<f64>>) -> Self {
        Self {
            solver: ILQRSolver::new(
                state_dim,
                control_dim,
                DMatrix::from_row_slice(Q.len(), Q[0].len(), &Q.concat()),
                DMatrix::from_row_slice(R.len(), R[0].len(), &R.concat()),
            ),
        }
    }

    #[pyo3(signature = (x0, target, dynamics, time_steps, max_iterations=None, convergence_threshold=None))]
    fn solve(
        &self,
        x0: Vec<f64>,
        target: Vec<f64>,
        dynamics: Bound<'_, PyAny>,
        time_steps: usize,
        max_iterations: Option<usize>,
        convergence_threshold: Option<f64>,
    ) -> Vec<Vec<f64>> {
        // Check the input dimensions
        assert!(x0.len() == self.solver.state_dim, "Invalid state dimension");
        assert!(
            target.len() == self.solver.state_dim,
            "Invalid target dimension"
        );
        let max_iterations = max_iterations.unwrap_or(100);
        let convergence_threshold = convergence_threshold.unwrap_or(1e-2);

        // Convert the input to nalgebra types
        let x0 = DVector::from_row_slice(&x0);
        let target = DVector::from_row_slice(&target);

        // Solve the problem and convert the output
        let us = self.solver.solve(
            x0,
            target,
            |x, u| {
                dynamics
                    .call1((x, u))
                    .unwrap()
                    .extract::<Vec<f64>>()
                    .unwrap()
                    .into()
            },
            time_steps,
            max_iterations,
            convergence_threshold,
        );
        us.into_iter().map(|x| x.as_slice().to_vec()).collect()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.solver))
    }
}
