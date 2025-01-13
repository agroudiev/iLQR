use nalgebra::{DMatrix, DVector};
use numpy::ToPyArray;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

use crate::py_utils::{build_matrix_list, build_vector_list};

#[pyclass(frozen, extends=PyRuntimeError)]
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct InstableError {
    pub iteration: usize,
    pub control: PyObject,
    pub out_state: PyObject,
    pub xs: PyObject,
    pub Ks: PyObject,
    pub ds: PyObject,
}

impl<'py> IntoPyObject<'py> for InstableError {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        py.get_type::<InstableError>().call1((
            self.iteration,
            self.control,
            self.out_state,
            self.xs,
            self.Ks,
            self.ds,
        ))
    }
}

impl InstableError {
    #[allow(non_snake_case)]
    pub fn from_error(
        py: Python<'_>,
        iteration: usize,
        control: Vec<DVector<f64>>,
        out_state: Vec<DVector<f64>>,
        xs: Vec<DVector<f64>>,
        Ks: Vec<DMatrix<f64>>,
        ds: Vec<DVector<f64>>,
    ) -> InstableError {
        Self {
            iteration: iteration,
            control: build_vector_list(control)
                .to_pyarray(py)
                .into_any()
                .unbind(),
            out_state: build_vector_list(out_state)
                .to_pyarray(py)
                .into_any()
                .unbind(),
            xs: build_vector_list(xs).to_pyarray(py).into_any().unbind(),
            Ks: build_matrix_list(Ks).to_pyarray(py).into_any().unbind(),
            ds: build_vector_list(ds).to_pyarray(py).into_any().unbind(),
        }
    }
}

#[pymethods]
#[allow(non_snake_case)]
impl InstableError {
    #[new]
    fn new(
        it: usize,
        control: Bound<'_, PyAny>,
        out_state: Bound<'_, PyAny>,
        xs: Bound<'_, PyAny>,
        Ks: Bound<'_, PyAny>,
        ds: Bound<'_, PyAny>,
    ) -> InstableError {
        InstableError {
            iteration: it,
            control: control.unbind(),
            out_state: out_state.unbind(),
            xs: xs.unbind(),
            Ks: Ks.unbind(),
            ds: ds.unbind(),
        }
    }

    #[getter]
    fn control(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.control.clone_ref(py))
    }

    #[getter]
    fn out_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.out_state.clone_ref(py))
    }

    #[getter]
    fn xs(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.xs.clone_ref(py))
    }

    #[getter]
    fn Ks(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.Ks.clone_ref(py))
    }

    #[getter]
    fn ds(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.ds.clone_ref(py))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "Instable problem - NaN detected in the control sequence. Choose a small gradient clipping value, or reduce the number of iterations. (iteration: {})", self.iteration
        ))
    }
}
