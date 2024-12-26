use crate::ilqr_solver::ILQRSolver;
use nalgebra::{DMatrix, DVector};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

const MAX_ITERATIONS: usize = 100;
const CONVERGENCE_THRESHOLD: f64 = 1e-2;
const VERBOSE: bool = false;
const GRADIENT_CLIP: f64 = 10.0;

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
        Q: Bound<PyAny>,
        Qf: Bound<PyAny>,
        R: Bound<PyAny>,
    ) -> Self {
        // TODO: clean error handling for wrong shapes
        // Convert the numpy arrays to Rust types
        let Q = Q.extract::<Vec<Vec<f64>>>().unwrap();
        let Qf = Qf.extract::<Vec<Vec<f64>>>().unwrap();
        let R = R.extract::<Vec<Vec<f64>>>().unwrap();

        assert!(Q.len() == state_dim, "Invalid state cost matrix dimension");
        assert!(
            Q[0].len() == state_dim,
            "Invalid state cost matrix dimension"
        );
        assert!(
            Qf.len() == state_dim,
            "Invalid final state cost matrix dimension"
        );
        assert!(
            Qf[0].len() == state_dim,
            "Invalid final state cost matrix dimension"
        );
        assert!(
            R.len() == control_dim,
            "Invalid control cost matrix dimension"
        );
        assert!(
            R[0].len() == control_dim,
            "Invalid control cost matrix dimension"
        );

        Self {
            solver: ILQRSolver::new(
                state_dim,
                control_dim,
                DMatrix::from_row_slice(Q.len(), Q[0].len(), &Q.concat()),
                DMatrix::from_row_slice(Qf.len(), Qf[0].len(), &Qf.concat()),
                DMatrix::from_row_slice(R.len(), R[0].len(), &R.concat()),
            ),
        }
    }

    #[pyo3(signature = (x0, target, dynamics, time_steps, max_iterations=None, convergence_threshold=None, gradient_clip=None, verbose=None))]
    fn solve(
        &self,
        py: Python<'_>,
        x0: Bound<PyAny>,
        target: Bound<PyAny>,
        dynamics: Bound<'_, PyAny>,
        time_steps: usize,
        max_iterations: Option<usize>,
        convergence_threshold: Option<f64>,
        gradient_clip: Option<f64>,
        verbose: Option<bool>,
    ) -> Py<PyArray2<f64>> {
        // TODO: clean error handling for wrong shapes
        let x0 = x0.extract::<Vec<f64>>().unwrap();
        let target = target.extract::<Vec<f64>>().unwrap();

        // Check the input dimensions
        assert!(
            x0.len() == self.solver.state_dim,
            "Invalid initial state dimension; expected {} but got {}",
            self.solver.state_dim,
            x0.len()
        );
        assert!(
            target.len() == self.solver.state_dim,
            "Invalid target dimension"
        );
        assert!(time_steps > 0, "Time steps must non-zero");
        let max_iterations = max_iterations.unwrap_or(MAX_ITERATIONS);
        let convergence_threshold = convergence_threshold.unwrap_or(CONVERGENCE_THRESHOLD);
        let verbose = verbose.unwrap_or(VERBOSE);
        let gradient_clip = gradient_clip.unwrap_or(GRADIENT_CLIP);

        // Convert the input to nalgebra types
        let x0 = DVector::from_row_slice(&x0);
        let target = DVector::from_row_slice(&target);

        // Solve the problem
        let us = self.solver.solve(
            x0,
            target,
            |x, u| {
                dynamics
                    .call1((x.to_pyarray(py), u.to_pyarray(py)))
                    .unwrap()
                    .extract::<Vec<f64>>()
                    .unwrap()
                    .into()
            },
            time_steps,
            max_iterations,
            convergence_threshold,
            gradient_clip,
            verbose,
        );

        // TODO: optimize the conversion
        // Convert Vec<DVector<f64>> to Vec<Vec<f64>>
        let us = us
            .into_iter()
            .map(|x| x.as_slice().to_vec())
            .collect::<Vec<Vec<f64>>>();

        // convert Vec<Vec<f64>> to Array2<f64>
        Array2::from_shape_fn((us.len(), us[0].len()), |(i, j)| us[i][j])
            .into_pyarray(py)
            .into()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.solver))
    }
}
