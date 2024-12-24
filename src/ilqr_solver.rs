use nalgebra::{DMatrix, DVector};
use pyo3::pyclass;

#[allow(non_snake_case)]
#[pyclass]
#[derive(Debug)]
pub struct ILQRSolver {
    /// Dimension of the state space
    pub state_dim: usize,
    /// Dimension of the control space
    pub control_dim: usize,
    /// State cost matrix
    pub Q: DMatrix<f64>,
    /// Control cost matrix
    pub R: DMatrix<f64>,
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

    /// Computes the Jacobians of the dynamics with respect to the state and control,
    /// at state `x` and control `u`
    ///
    /// * `x` - The current state
    /// * `u` - The control
    ///
    /// Returns: `(A, B)` where A = ∂f/∂x and B = ∂f/∂u at `(x, u)`
    fn linearize_dynamics(&self, x: DVector<f64>, u: DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        unimplemented!()
    }

    /// Computes the forward pass from a given state `x`, controls `us`, and a `target`
    ///
    /// * `x` - The current state
    /// * `us` - The control sequence
    /// * `target` - The target state
    ///
    /// Returns: a tuple `(xs, loss)` containing the states and loss
    fn forward(
        &self,
        x: &DVector<f64>,
        us: &Vec<DVector<f64>>,
        target: &DVector<f64>,
    ) -> (Vec<DVector<f64>>, f64) {
        unimplemented!()
    }

    /// Computes the backward pass from a given state `x`, controls `us`, and a `target`
    ///
    /// * `x` - The current state
    /// * `us` - The control sequence
    /// * `target` - The target state
    ///
    /// Returns: a tuple `(Ks, ds)` containing the control gains and the forcing gains
    fn backward(
        &self,
        xs: &Vec<DVector<f64>>,
        us: &Vec<DVector<f64>>,
        target: &DVector<f64>,
    ) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        unimplemented!()
    }

    /// Solves the ILQR problem from a given initial state `x0` and target `target`
    ///
    /// * `x0` - The initial state
    /// * `target` - The target state
    /// * `dynamics` - The dynamics function, that computes the next state given the current state `x` and control `u`
    ///
    /// Returns: the sequence of controls
    pub fn solve(
        &self,
        x0: DVector<f64>,
        target: DVector<f64>,
        dynamics: impl Fn(&[f64], &[f64]) -> DVector<f64>,
        time_steps: usize,
        max_iterations: usize,
        convergence_threshold: f64,
    ) -> Vec<DVector<f64>> {
        // Initialize the trajectory
        let mut us = vec![DVector::zeros(self.control_dim); time_steps];
        // TODO: implement different initialization strategies

        for _ in 0..max_iterations {
            // Forward pass
            let (mut xs, _loss) = self.forward(&x0, &us, &target);
            // Backward pass
            #[allow(non_snake_case)]
            let (Ks, ds) = self.backward(&xs, &us, &target);

            // Update the controls
            let mut x = x0.clone();
            let mut dus: Vec<DVector<f64>> = vec![DVector::zeros(self.control_dim); time_steps];
            for i in 0..time_steps {
                let du = &Ks[i] * (&x - &xs[i]) + &ds[i];

                // TODO: gradient clip `du`

                us[i] += &du;
                dus[i] = du;

                x = dynamics(x.as_slice(), us[i].as_slice());
                xs[i] = x.clone();
            }

            // Check for convergence
            let norm = dus.iter().map(|du| du.norm()).sum::<f64>().sqrt();
            if norm < convergence_threshold {
                break;
            }
        }

        us
    }
}
