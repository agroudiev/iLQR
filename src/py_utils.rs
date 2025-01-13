use crate::ilqr_solver::ILQRError;
use crate::py_ilqr_errors::InstableError;
use nalgebra::{DMatrix, DVector};
use numpy::ndarray::{Array2, Array3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub fn build_vector_list(data: Vec<DVector<f64>>) -> Array2<f64> {
    let nb_vec = data.len();
    let vec_size = data[0].len();

    // Flatten the data
    let flat_data: Vec<f64> = data
        .into_iter()
        .flat_map(|x| x.data.as_vec().clone())
        .collect::<Vec<f64>>();

    // And reshape it to expected size
    Array2::from_shape_vec((nb_vec, vec_size), flat_data).unwrap()
}

pub fn build_matrix_list(data: Vec<DMatrix<f64>>) -> Array3<f64> {
    let nb_vec = data.len();
    let (mat_rows, mat_cols) = data[0].shape();

    // Flatten the data
    let flat_data: Vec<f64> = data
        .into_iter()
        .flat_map(|x| x.data.as_vec().clone())
        .collect::<Vec<f64>>();

    // And reshape it to expected size
    Array3::from_shape_vec((nb_vec, mat_rows, mat_cols), flat_data).unwrap()
}

pub fn build_error(py: Python<'_>, err: ILQRError<PyErr>) -> PyErr {
    match err {
        ILQRError::QUUNotInvertible => {
            PyRuntimeError::new_err("The Quu matrix should be invertible.")
        }
        ILQRError::InstableProblem {
            iteration,
            control,
            out_state,
            xs,
            Ks,
            ds,
        } => {
            let error = InstableError::from_error(py, iteration, control, out_state, xs, Ks, ds);

            match error.into_pyobject(py) {
                Ok(error) => PyErr::from_value(error),
                Err(err) => err,
            }
        }
        ILQRError::DynamicsError(e) => e,
    }
}
