use std::time::Instant;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand_distr::{Distribution, Normal};

/// The epsilon used for finite differences
const EPSILON: f64 = 1e-5;
/// The epsilon used for eigendecomposition
const EIGENDECOMP_EPSILON: f64 = 1e-5;

/// Decrease and Increase factor of the regularisation term
const REG_FACTOR: f64 = 2.0;
/// Maximum value of the regularisation term
const REG_MAX: f64 = 1e3;
/// Minimum value of the regularisation term
const REG_MIN: f64 = 1e-5;

#[derive(Debug)]
#[allow(non_snake_case)]
pub enum ILQRError<E> {
    QUUNotInvertible,
    InstableProblem {
        /// Iteration of the Nan occurrence
        iteration: usize,
        /// Last Control Sequence
        control: Vec<DVector<f64>>,
        /// Last State Sequence
        out_state: Vec<DVector<f64>>,
        /// Last Forward Pass
        xs: Vec<DVector<f64>>,
        /// Last Control Gains
        Ks: Vec<DMatrix<f64>>,
        /// Last Forcing Gains
        ds: Vec<DVector<f64>>,
    },
    DynamicsError(E),
}

type ILQRResult<T, E> = Result<T, ILQRError<E>>;

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct ILQRSolver {
    /// Dimension of the state space
    pub state_dim: usize,
    /// Dimension of the control space
    pub control_dim: usize,
    /// State cost matrix
    pub Q: DMatrix<f64>,
    /// Final state cost matrix
    pub Qf: DMatrix<f64>,
    /// Control cost matrix
    pub R: DMatrix<f64>,
}

#[derive(Debug)]
pub enum OutputKind {
    Partial,
    Converged,
}

#[derive(Debug)]
pub struct ILQROutput {
    pub kind: OutputKind,
    pub control: Vec<DVector<f64>>,
    pub time_taken: f64,
    pub it_taken: usize,
    pub cost: Vec<f64>,
    pub gradient_norm: Vec<f64>,
    pub jac_time_taken: Vec<f64>,
}

pub enum ILQRStopThreshold {
    GradientNormThreshold(f64),
    CostThreshold(f64),
}

impl ILQRSolver {
    /// Creates a new ILQRSolver.
    ///
    /// * `state_dim` - Dimension of the state space
    /// * `control_dim` - Dimension of the control space
    /// * `Q` - State cost matrix
    /// * `Qf` - Final state cost matrix
    /// * `R` - Control cost matrix
    #[allow(non_snake_case)]
    pub fn new(
        state_dim: usize,
        control_dim: usize,
        Q: DMatrix<f64>,
        Qf: DMatrix<f64>,
        R: DMatrix<f64>,
    ) -> Self {
        Self {
            state_dim,
            control_dim,
            Q,
            Qf,
            R,
        }
    }

    /// Computes the Jacobians of the dynamics with respect to the state and control,
    /// at state `x` and control `u`. Uses the finite difference method.
    ///
    /// * `f` - The function to linearize (dynamics)
    /// * `x` - The current state
    /// * `u` - The control
    ///
    /// Returns: `(A, B)` where A = ∂f/∂x and B = ∂f/∂u at `(x, u)`
    fn jacobians<E, D, J>(
        &self,
        x: DVector<f64>,
        u: DVector<f64>,
        dynamics: &D,
        jac: &Option<J>,
    ) -> ILQRResult<(DMatrix<f64>, DMatrix<f64>, f64), E>
    where
        D: Fn(&[f64], &[f64]) -> Result<DVector<f64>, E>,
        J: Fn(&[f64], &[f64]) -> Result<(DMatrix<f64>, DMatrix<f64>), E>,
    {
        if let Some(j) = jac {
            let beg: Instant = Instant::now();

            let (jac_x, jac_u) = j(x.as_slice(), u.as_slice()).map_err(ILQRError::DynamicsError)?;

            let end: Instant = Instant::now();

            return Ok((jac_x, jac_u, (end - beg).as_secs_f64()));
        }

        #[allow(non_snake_case)]
        let mut A = DMatrix::zeros(self.state_dim, self.state_dim);
        #[allow(non_snake_case)]
        let mut B = DMatrix::zeros(self.state_dim, self.control_dim);

        let beg: Instant = Instant::now();

        for i in 0..self.state_dim {
            let mut xpdx = x.clone();
            xpdx[i] += EPSILON;
            let mut xmdx = x.clone();
            xmdx[i] -= EPSILON;

            let f_xpdx =
                dynamics(xpdx.as_slice(), u.as_slice()).map_err(ILQRError::DynamicsError)?;
            let f_xmdx =
                dynamics(xmdx.as_slice(), u.as_slice()).map_err(ILQRError::DynamicsError)?;

            let df_dx = (f_xpdx - f_xmdx) / (2.0 * EPSILON);
            A.set_column(i, &df_dx);
        }

        for i in 0..self.control_dim {
            let mut updu = u.clone();
            updu[i] += EPSILON;
            let mut umdu = u.clone();
            umdu[i] -= EPSILON;

            let f_updu =
                dynamics(x.as_slice(), updu.as_slice()).map_err(ILQRError::DynamicsError)?;
            let f_umdu =
                dynamics(x.as_slice(), umdu.as_slice()).map_err(ILQRError::DynamicsError)?;

            let df_du = (f_updu - f_umdu) / (2.0 * EPSILON);
            B.set_column(i, &df_du);
        }
        let end: Instant = Instant::now();

        Ok((A, B, (end - beg).as_secs_f64()))
    }

    /// Computes the LQR forward pass from a given state `x`, controls `us`, and a `target`
    ///
    /// * `x` - The current state
    /// * `us` - The control sequence
    /// * `target` - The target state
    ///
    /// Returns: a tuple `(xs, loss)` containing the states and loss
    fn forward<E, D>(
        &self,
        x: &DVector<f64>,
        us: &Vec<DVector<f64>>,
        target: &DVector<f64>,
        dynamics: &D,
    ) -> ILQRResult<(Vec<DVector<f64>>, f64), E>
    where
        D: Fn(&[f64], &[f64]) -> Result<DVector<f64>, E>,
    {
        let mut x = x.clone();
        let mut xs = vec![x.clone()];
        let mut loss = 0.0;

        for u in us {
            x = dynamics(x.as_slice(), u.as_slice()).map_err(ILQRError::DynamicsError)?;
            let delta = &x - target;
            loss += ((delta.transpose() * &self.Q).dot(&delta.transpose()))
                + (&u.transpose() * &self.R).transpose().dot(u);
            xs.push(x.clone());
        }

        let delta = &x - target;
        loss += (delta.transpose() * &self.Qf).dot(&delta.transpose());

        Ok((xs, loss))
    }

    /// Computes the LQR backward pass given the states `xs`, controls `us`, and a `target`
    ///
    /// * `x` - The current state
    /// * `us` - The control sequence
    /// * `target` - The target state
    /// * `dynamics` - The dynamics function
    ///
    /// Returns: On success, a tuple `(Ks, ds)` containing the control gains and the forcing gains.
    #[allow(non_snake_case)]
    fn backward<E, D, J>(
        &self,
        xs: &[DVector<f64>],
        us: &[DVector<f64>],
        target: &DVector<f64>,
        dynamics: &D,
        jac: &Option<J>,
        use_regulation: bool,
        lamb: f64,
    ) -> ILQRResult<(Vec<DMatrix<f64>>, Vec<DVector<f64>>, f64), E>
    where
        D: Fn(&[f64], &[f64]) -> Result<DVector<f64>, E>,
        J: Fn(&[f64], &[f64]) -> Result<(DMatrix<f64>, DMatrix<f64>), E>,
    {
        let mut Ks = vec![DMatrix::zeros(self.control_dim, self.state_dim); xs.len()];
        let mut ds = vec![DVector::zeros(self.control_dim); xs.len()];

        let mut s = self.Qf.clone() * (xs.last().unwrap() - target);
        let mut S = self.Qf.clone();

        let mut jac_time = 0.0;

        for i in (0..xs.len() - 1).rev() {
            let x = &xs[i];
            let u = &us[i];

            let (A, B, jac_timing) = self.jacobians(x.clone(), u.clone(), &dynamics, &jac)?;
            jac_time += jac_timing;

            let Qx = &self.Q * &(x - target) + (&s.transpose() * &A).transpose();
            let Qu = &self.R * u + (&s.transpose() * &B).transpose();

            let Qxx = &self.Q + A.transpose() * &S * &A;
            let Quu = &self.R + B.transpose() * &S * &B;
            let Qux = &B.transpose() * &S * &A;

            let Quu_inv = if use_regulation {
                let mut decomp = SymmetricEigen::try_new(Quu.clone(), EIGENDECOMP_EPSILON, 0)
                    .map_or(Err(ILQRError::QUUNotInvertible), Ok)?;

                decomp.eigenvalues.apply(|x| {
                    if *x <= 0.0 {
                        *x = 0.0;
                    } else {
                        *x = 1.0 / (*x + lamb); // Inversion and regularization term
                    }
                });

                decomp.recompose()
            } else {
                Quu.clone()
                    .try_inverse()
                    .map_or_else(|| Err(ILQRError::QUUNotInvertible), Ok)?
            };

            Ks[i] = -&Quu_inv * &Qux;
            ds[i] = -&Quu_inv * &Qu;

            s = Qx;
            s += Ks[i].transpose() * &Quu * &ds[i];
            s += Ks[i].transpose() * Qu;
            s += Qux.transpose() * &ds[i];

            S = Qxx;
            S += Ks[i].transpose() * &Quu * &Ks[i];
            S += Ks[i].transpose() * &Qux;
            S += Qux.transpose() * &Ks[i];
        }

        let jac_avg_time = jac_time / us.len() as f64;

        Ok((Ks, ds, jac_avg_time))
    }

    /// Solves the ILQR problem from a given initial state `x0` and target `target`
    ///
    /// * `x0` - The initial state
    /// * `target` - The target state
    /// * `dynamics` - The dynamics function, that computes the next state given the current state `x` and control `u`
    ///
    /// Returns: On success, the sequence of controls
    pub fn solve<E, D, J, C>(
        &self,
        x0: DVector<f64>,
        target: DVector<f64>,
        dynamics: D,
        time_steps: usize,
        callback: C,
        initialization: f64,
        max_iterations: usize,
        convergence_threshold: ILQRStopThreshold,
        jac: Option<J>,
        gradient_clip: Option<f64>,
        use_regulation: bool,
        verbose: bool,
        warmstart: Option<Vec<DVector<f64>>>,
    ) -> ILQRResult<ILQROutput, E>
    where
        D: Fn(&[f64], &[f64]) -> Result<DVector<f64>, E>,
        J: Fn(&[f64], &[f64]) -> Result<(DMatrix<f64>, DMatrix<f64>), E>,
        C: Fn(f64) -> (),
    {
        // Initialize cost and norm vectors
        let mut cost = Vec::new();
        let mut norm = Vec::new();
        let mut jac_time = Vec::new();

        // Initialize the trajectory
        let mut us = if let Some(warmstart) = warmstart {
            // Initialize the controls to the warmstart
            warmstart
        }
        else if initialization == 0.0 {
            // Initialize the controls to zeros
            vec![DVector::zeros(self.control_dim); time_steps]
        } else {
            // Initialize the controls to random values from a normal distribution of mean 0 and std `initialization`
            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, initialization).unwrap();

            let mut us = Vec::with_capacity(time_steps);
            for _ in 0..time_steps {
                let u = DVector::from_fn(self.control_dim, |_, _| normal.sample(&mut rng));
                us.push(u);
            }
            us
        };

        let mut dus: Vec<DVector<f64>> = vec![DVector::zeros(self.control_dim); time_steps];
        let mut x_sim: Vec<DVector<f64>> = vec![DVector::zeros(self.state_dim); time_steps + 1];

        x_sim[0] = x0.clone();

        let mut prev_cost: Option<f64> = None;
        let mut reg_coef = 1.0; // dynamic regularisation term

        if verbose {
            println!("==== Starting iLQR ====");
            println!(
                "Initial controls: {:?} (generated with std {initialization})",
                us
            );
            println!("Initial state: {x0}");
            println!("Target state: {target}");
            println!("Gradient clip: {:?}", gradient_clip);
        }

        let beg: Instant = Instant::now();

        for iteration in 0..max_iterations {
            if verbose {
                println!("\n== Iteration: {iteration} ==");
            }
            // Forward pass
            let (xs, loss) = self.forward(&x0, &us, &target, &dynamics)?;

            if verbose {
                println!("Loss: {loss}");
            }

            // Backward pass
            #[allow(non_snake_case)]
            let (Ks, ds, jac_timing) =
                self.backward(&xs, &us, &target, &dynamics, &jac, use_regulation, reg_coef)?;
            jac_time.push(jac_timing);

            // Update the controls
            let mut cost_i = 0.0;
            for i in 0..time_steps {
                // Do it at the beginning to compute the sum from 0 to N-1
                cost_i += (&x_sim[i].transpose() * &self.Q * &x_sim[i]).as_scalar()
                    + (&us[i].transpose() * &self.R * &us[i]).as_scalar();

                let mut du = &Ks[i] * (&x_sim[i] - &xs[i]) + &ds[i];

                match gradient_clip {
                    // gradient clip `du` if its norm is greater than `gradient_clip`
                    Some(gradient_clip) => {
                        let norm = du.norm();
                        if norm > gradient_clip {
                            du = du / norm * gradient_clip;
                        }
                    }
                    _ => {}
                }

                us[i] += &du;
                dus[i] = du;

                x_sim[i + 1] = dynamics(x_sim[i].as_slice(), us[i].as_slice())
                    .map_err(ILQRError::DynamicsError)?;
            }
            // Add the final terms of the cost
            let x_diff = &x_sim[time_steps] - &target;
            cost_i += (x_diff.transpose() * &self.Qf * x_diff).as_scalar();

            // The squared norm is the sum of the square of all elements
            let norm_i = dus
                .iter()
                .map(|du| du.norm() * du.norm())
                .sum::<f64>()
                .sqrt();

            cost.push(cost_i);
            norm.push(norm_i);
            callback(cost_i);

            if verbose {
                println!("Gradient norm: {norm_i}");
                println!("Cost: {cost_i}");
            }

            if use_regulation {
                if prev_cost.is_none() || cost_i < prev_cost.unwrap() {
                    reg_coef = f64::max(reg_coef / REG_FACTOR, REG_MIN);
                } else {
                    reg_coef = f64::min(reg_coef * REG_FACTOR, REG_MAX);
                }
                prev_cost = Some(cost_i);
            }

            // Check for convergence
            let converged = match convergence_threshold {
                ILQRStopThreshold::GradientNormThreshold(norm_thresh) => norm_i < norm_thresh,
                ILQRStopThreshold::CostThreshold(cost_thresh) => cost_i < cost_thresh,
            };

            if converged {
                let end = Instant::now();

                if verbose {
                    println!("==== Break: Converged ====");
                }
                return Ok(ILQROutput {
                    kind: OutputKind::Converged,
                    control: us,
                    time_taken: (end - beg).as_secs_f64(),
                    it_taken: iteration,
                    cost: cost,
                    gradient_norm: norm,
                    jac_time_taken: jac_time,
                });
            }

            if norm_i.is_nan() {
                return Err(ILQRError::InstableProblem {
                    iteration,
                    control: us,
                    out_state: x_sim,
                    xs,
                    Ks,
                    ds,
                });
            }
        }
        let end: Instant = Instant::now();

        return Ok(ILQROutput {
            kind: OutputKind::Partial,
            control: us,
            time_taken: (end - beg).as_secs_f64(),
            it_taken: max_iterations,
            cost: cost,
            gradient_norm: norm,
            jac_time_taken: jac_time,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(non_snake_case)]
    fn test_ilqr_solver(
        state_dim: usize,
        control_dim: usize,
        initialization: f64,
        gradient_clip: Option<f64>,
    ) {
        let Q = DMatrix::identity(state_dim, state_dim);
        let Qf = DMatrix::identity(state_dim, state_dim);
        let R = DMatrix::identity(control_dim, control_dim);

        let solver = ILQRSolver::new(state_dim, control_dim, Q, Qf, R);

        let x0 = DVector::zeros(state_dim);
        let target = DVector::from_element(state_dim, 1.0);

        let dynamics = |x: &[f64], _u: &[f64]| Ok::<DVector<f64>, ()>(DVector::from_row_slice(&x));

        let callback = |_: f64| {};

        let _ = solver.solve(
            x0,
            target,
            dynamics,
            10,
            callback,
            initialization,
            100,
            ILQRStopThreshold::GradientNormThreshold(1e-2),
            None::<fn(&[f64], &[f64]) -> Result<(DMatrix<f64>, DMatrix<f64>), ()>>,
            gradient_clip,
            false,
            false,
            None
        );
    }

    #[test]
    fn test_simple() {
        test_ilqr_solver(2, 1, 0.0, None);
    }

    #[test]
    fn test_simple_random() {
        test_ilqr_solver(2, 1, 1.0, None);
    }

    #[test]
    fn test_complex() {
        test_ilqr_solver(4, 2, 0.0, Some(1.0));
    }

    #[test]
    fn test_complex_random() {
        test_ilqr_solver(4, 2, 1.0, Some(1.0));
    }
}
