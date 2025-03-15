use crate::ilqr_solver::{ILQRSolver, ILQRStopThreshold, OutputKind};
use crate::py_utils::{build_error, build_vector_list};
use nalgebra::{DMatrix, DVector};
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{pyclass, pymethods};

/// Maximum number of iterations
const MAX_ITERATIONS: usize = 100;
/// Threshold
const DEFAULT_THRESHOLD: ILQRStopThreshold = ILQRStopThreshold::GradientNormThreshold(1e-2);
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

fn check_mat(mat: Vec<Vec<f64>>, nrows: usize, ncols: usize, msg: &str) -> PyResult<DMatrix<f64>> {
    if mat.len() != nrows || mat[0].len() != ncols {
        return Err(PyValueError::new_err(format!(
            "{msg}; expected {nrows}x{ncols} but got {}x{}.",
            mat.len(),
            mat[0].len()
        )));
    }
    return Ok(DMatrix::from_row_slice(nrows, ncols, &mat.concat()));
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

        let Q = check_mat(
            Q,
            state_dim,
            state_dim,
            "Invalid state cost matrix dimension",
        )?;

        let Qf = check_mat(
            Qf,
            state_dim,
            state_dim,
            "Invalid final state cost matrix dimension",
        )?;
        let R = check_mat(
            R,
            control_dim,
            control_dim,
            "Invalid control cost matrix dimension",
        )?;

        Ok(Self {
            solver: ILQRSolver::new(state_dim, control_dim, Q, Qf, R),
        })
    }

    #[pyo3(signature = (x0, target, dynamics, time_steps, callback, jacobians=None, initialization=None, max_iterations=None, cost_threshold=None, gradient_threshold=None, gradient_clip=None, regularize=None, verbose=None, full_output=None, warmstart=None))]
    fn solve(
        &self,
        py: Python<'_>,
        x0: Bound<PyAny>,
        target: Bound<PyAny>,
        dynamics: Bound<'_, PyAny>,
        time_steps: usize,
        callback: Bound<'_, PyAny>,
        jacobians: Option<Bound<'_, PyAny>>,
        initialization: Option<f64>,
        max_iterations: Option<usize>,
        cost_threshold: Option<f64>,
        gradient_threshold: Option<f64>,
        gradient_clip: Option<f64>,
        regularize: Option<bool>,
        verbose: Option<bool>,
        full_output: Option<bool>,
        warmstart: Option<Bound<PyAny>>
    ) -> PyResult<Py<PyAny>> {
        let x0 = x0.extract::<Vec<f64>>()?;
        let target = target.extract::<Vec<f64>>()?;

        let max_iterations = max_iterations.unwrap_or(MAX_ITERATIONS);
        let threshold = gradient_threshold.map_or(
            cost_threshold.map_or(DEFAULT_THRESHOLD, ILQRStopThreshold::CostThreshold),
            ILQRStopThreshold::GradientNormThreshold,
        );
        let verbose = verbose.unwrap_or(VERBOSE);
        // let gradient_clip = gradient_clip.unwrap_or(GRADIENT_CLIP);

        let warmstart = match warmstart {
            Some(warmstart) => {
                let warmstart = warmstart.extract::<Vec<f64>>()?;
                assert_eq!(warmstart.len(), self.solver.control_dim * time_steps);
                let mut warmstart_vec = vec![];

                for chunk in warmstart.chunks(self.solver.control_dim) {
                    warmstart_vec.push(DVector::from_row_slice(chunk));
                }
                Some(warmstart_vec)
            },
            None => None
        };
        let initialization = initialization.unwrap_or(INITIALIZATION);
        let regularize = regularize.unwrap_or(false);
        let full_output = full_output.unwrap_or(false);

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

        let dyn_f = |x: &[f64], u: &[f64]| {
            Ok(dynamics
                .call1((x.to_pyarray(py), u.to_pyarray(py)))?
                .extract::<Vec<f64>>()?
                .into())
        };

        let callback = |cost: f64| {
            let _ = callback.call1((cost,));
        };

        let jac_f = if let Some(f) = &jacobians {
            Some(|x: &[f64], u: &[f64]| {
                f.call1((x.to_pyarray(py), u.to_pyarray(py)))
                    .and_then(|res| {
                        res.extract::<(Vec<Vec<f64>>, Vec<Vec<f64>>)>().and_then(
                            |(jac_x, jac_u)| {
                                let mat_j_x = check_mat(
                                    jac_x,
                                    self.solver.state_dim,
                                    self.solver.state_dim,
                                    "Invalid Jacobian size",
                                );
                                let mat_j_u = check_mat(
                                    jac_u,
                                    self.solver.state_dim,
                                    self.solver.control_dim,
                                    "Invalid Jacobian size",
                                );

                                match (mat_j_x, mat_j_u) {
                                    (Ok(mat_j_x), Ok(mat_j_u)) => Ok((mat_j_x, mat_j_u)),
                                    (Err(l), _) | (_, Err(l)) => Err(l),
                                }
                            },
                        )
                    })
            })
        } else {
            None
        };

        // Solve the problem
        let us = self
            .solver
            .solve(
                x0,
                target,
                dyn_f,
                time_steps,
                callback,
                initialization,
                max_iterations,
                threshold,
                jac_f,
                gradient_clip,
                regularize,
                verbose,
                warmstart
            )
            .map_err(|err| build_error(py, err))?;

        let np_array = build_vector_list(us.control).to_pyarray(py).into_any();

        if full_output {
            let output_struct = PyDict::new(py);
            output_struct.set_item("control", np_array)?;
            output_struct.set_item("time_taken", us.time_taken)?;
            output_struct.set_item("it_taken", us.it_taken)?;
            output_struct.set_item("gradient_norm", us.gradient_norm)?;
            output_struct.set_item("cost", us.cost)?;
            output_struct.set_item("jac_time_taken", us.jac_time_taken)?;

            match us.kind {
                OutputKind::Partial => output_struct.set_item("kind", "partial")?,
                OutputKind::Converged => output_struct.set_item("kind", "converged")?,
            }

            Ok(output_struct.into_any().unbind())
        } else {
            Ok(np_array.unbind())
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.solver))
    }
}
