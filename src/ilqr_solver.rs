use nalgebra::{DMatrix, DVector};
use pyo3::pyclass;

#[allow(non_snake_case)]
#[pyclass]
#[derive(Debug)]
pub struct ILQRSolver {
    /// Dimension of the state space
    state_dim: usize,
    /// Dimension of the control space
    control_dim: usize,
    // /// Compute the next state given the current state and control
    // dynamics: std::sync::Arc<dyn Fn(&[f64], &[f64]) -> Vec<f64>>,
    /// State cost matrix
    Q: DMatrix<f64>,
    /// Control cost matrix
    R: DMatrix<f64>,
}

impl ILQRSolver {
    /// Create a new ILQRSolver
    ///
    /// * `state_dim` - Dimension of the state space
    /// * `control_dim` - Dimension of the control space
    /// * `dynamics` - Compute the next state given the current state `x` and control `u`
    /// * `Q` - State cost matrix
    /// * `R` - Control cost matrix
    #[allow(non_snake_case)]
    pub fn new(
        state_dim: usize,
        control_dim: usize,
        // dynamics: impl Fn(&[f64], &[f64]) -> Vec<f64>,
        Q: DMatrix<f64>,
        R: DMatrix<f64>,
    ) -> Self {
        Self {
            state_dim,
            control_dim,
            // dynamics,
            Q,
            R,
        }
    }

    /// Compute the Jacobians of the dynamics with respect to the state and control,
    /// at state `x` and control `u`
    ///
    /// * `x` - The current state
    /// * `u` - The control
    ///
    /// Returns: `(A, B)` where A = ∂f/∂x and B = ∂f/∂u at `(x, u)`
    fn linearize_dynamics(&self, x: DVector<f64>, u: DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        unimplemented!()
    }

    /// Compute the forward pass from a given state `x` and controls `us`
    ///
    /// * `x` - The current state
    /// * `us` - The control sequence
    ///
    /// Returns: a tuple containing the states and loss
    pub fn forward(&self, x: DVector<f64>, us: Vec<DVector<f64>>) -> (Vec<DMatrix<f64>>, f64) {
        unimplemented!()
    }

    /// Compute the backward pass from a given state `x`, controls `us`, and a target `target`
    ///
    /// * `x` - The current state
    /// * `us` - The control sequence
    /// * `target` - The target state
    ///
    /// Returns: a tuple containing the control gains and the forcing gains
    pub fn backward(
        &self,
        x: DVector<f64>,
        us: Vec<DVector<f64>>,
        target: DVector<f64>,
    ) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        unimplemented!()
    }
}
