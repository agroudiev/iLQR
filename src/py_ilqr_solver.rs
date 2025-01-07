use crate::ilqr_solver::{ILQRError, ILQRSolver};
use nalgebra::{DMatrix, DVector};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

/// Maximum number of iterations
const MAX_ITERATIONS: usize = 100;
/// Threshold for the gradient norm
const CONVERGENCE_THRESHOLD: f64 = 1e-2;
/// Verbosity flag
const VERBOSE: bool = false;
// /// Gradient clipping value
// const GRADIENT_CLIP: f64 = 10.0;
/// Standard deviation for the initialization
const INITIALIZATION: f64 = 0.0;

#[pyclass(name = "ILQRSolver")]
#[derive(Debug)]
/// A Python wrapper for `ILQRSolver`.
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
    ) -> PyResult<Self> {
        // Convert the numpy arrays to Rust types
        let Q = Q.extract::<Vec<Vec<f64>>>()?;
        let Qf = Qf.extract::<Vec<Vec<f64>>>()?;
        let R = R.extract::<Vec<Vec<f64>>>()?;

        if Q.len() != state_dim || Q[0].len() != state_dim {
            return Err(PyValueError::new_err(format!(
                "Invalid state cost matrix dimension; expected {state_dim}x{state_dim} but got {}x{}.",
                Q.len(),
                Q[0].len()
            )));
        }

        if Qf.len() != state_dim || Qf[0].len() != state_dim {
            return Err(PyValueError::new_err(format!(
                "Invalid final state cost matrix dimension; expected {state_dim}x{state_dim} but got {}x{}.",
                Qf.len(),
                Qf[0].len()
            )));
        }

        if R.len() != control_dim || R[0].len() != control_dim {
            return Err(PyValueError::new_err(format!(
                "Invalid control cost matrix dimension; expected {control_dim}x{control_dim} but got {}x{}.",
                R.len(),
                R[0].len()
            )));
        }

        Ok(Self {
            solver: ILQRSolver::new(
                state_dim,
                control_dim,
                DMatrix::from_row_slice(Q.len(), Q[0].len(), &Q.concat()),
                DMatrix::from_row_slice(Qf.len(), Qf[0].len(), &Qf.concat()),
                DMatrix::from_row_slice(R.len(), R[0].len(), &R.concat()),
            ),
        })
    }

    #[pyo3(signature = (x0, target, dynamics, time_steps, initialization=None, max_iterations=None, convergence_threshold=None, gradient_clip=None, verbose=None))]
    fn solve(
        &self,
        py: Python<'_>,
        x0: Bound<PyAny>,
        target: Bound<PyAny>,
        dynamics: Bound<'_, PyAny>,
        time_steps: usize,
        initialization: Option<f64>,
        max_iterations: Option<usize>,
        convergence_threshold: Option<f64>,
        gradient_clip: Option<f64>,
        verbose: Option<bool>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let x0 = x0.extract::<Vec<f64>>()?;
        let target = target.extract::<Vec<f64>>()?;

        let max_iterations = max_iterations.unwrap_or(MAX_ITERATIONS);
        let convergence_threshold = convergence_threshold.unwrap_or(CONVERGENCE_THRESHOLD);
        let verbose = verbose.unwrap_or(VERBOSE);
        // let gradient_clip = gradient_clip.unwrap_or(GRADIENT_CLIP);
        let initialization = initialization.unwrap_or(INITIALIZATION);

        // Check the input dimensions
        if x0.len() != self.solver.state_dim {
            return Err(PyValueError::new_err(format!(
                "Invalid initial state dimension; expected {} but got {}.",
                self.solver.state_dim,
                x0.len()
            )));
        }

        if target.len() != self.solver.state_dim {
            return Err(PyValueError::new_err(format!(
                "Invalid target dimension; expected {} but got {}.",
                self.solver.state_dim,
                target.len()
            )));
        }
        if time_steps == 0 {
            return Err(PyValueError::new_err("Time steps must non-zero."));
        }
        if initialization < 0.0 {
            return Err(PyValueError::new_err(
                "Initialization standard deviation must be non-negative.",
            ));
        }

        // Convert the input to nalgebra types
        let x0 = DVector::from_row_slice(&x0);
        let target = DVector::from_row_slice(&target);

        // Solve the problem
        let us = self.solver.solve(
            x0,
            target,
            |x, u| {
                Ok(dynamics
                    .call1((x.to_pyarray(py), u.to_pyarray(py)))?
                    .extract::<Vec<f64>>()?
                    .into())
            },
            time_steps,
            initialization,
            max_iterations,
            convergence_threshold,
            gradient_clip,
            verbose,
        )?;

        // Flatten the data
        let flat_data: Vec<f64> = us
            .into_iter()
            .flat_map(|x| x.data.as_vec().clone())
            .collect::<Vec<f64>>();

        // And reshape it to expected size
        Ok(
            Array2::from_shape_vec((time_steps, self.solver.control_dim), flat_data)
                .unwrap()
                .into_pyarray(py)
                .into(),
        )
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.solver))
    }
}

impl std::convert::From<ILQRError<PyErr>> for PyErr {
    fn from(err: ILQRError<PyErr>) -> PyErr {
        match err {
            ILQRError::QUUNotInvertible => PyRuntimeError::new_err("The Quu matrix should be invertible."),
            ILQRError::InstableProblem(iteration) =>
                PyRuntimeError::new_err(format!("Instable problem - NaN detected in the control sequence. Choose a small gradient clipping value, or reduce the number of iterations. (iteration: {iteration})")),
            ILQRError::DynamicsError(e) => e,
        }
    }
}
