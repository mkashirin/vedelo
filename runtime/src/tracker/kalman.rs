use crate::common::Rect;
use nalgebra::{SMatrix, SVector};

const STATE_DIM: usize = 7;
const MEAS_DIM: usize = 4;

type StateVec = SVector<f32, STATE_DIM>;
type MeasVec = SVector<f32, MEAS_DIM>;
type StateMat = SMatrix<f32, STATE_DIM, STATE_DIM>;
type MeasMat = SMatrix<f32, MEAS_DIM, STATE_DIM>;

pub struct KalmanFilter {
    state: StateVec,
    covariance: StateMat,
}

impl KalmanFilter {
    pub fn new(rect: Rect) -> Self {
        let x = rect.x as f32 + rect.width as f32 / 2.0;
        let y = rect.y as f32 + rect.height as f32 / 2.0;
        let area = (rect.width * rect.height) as f32;
        let ratio = rect.width as f32 / rect.height as f32;

        let mut state = StateVec::zeros();
        state[0] = x;
        state[1] = y;
        state[2] = area;
        state[3] = ratio;

        let mut covariance = StateMat::identity();
        for i in 0..7 {
            covariance[(i, i)] = if i < 4 { 10.0 } else { 1000.0 };
        }

        Self { state, covariance }
    }

    pub fn predict(&mut self) -> Rect {
        let mut f = StateMat::identity();
        f[(0, 4)] = 1.0;
        f[(1, 5)] = 1.0;
        f[(2, 6)] = 1.0;

        let mut q = StateMat::identity();
        for i in 0..7 {
            q[(i, i)] *= if i < 4 { 1.0 } else { 0.01 };
        }

        self.state = f * self.state;
        self.covariance = (f * self.covariance * f.transpose()) + q;

        self.state_to_rect()
    }

    pub fn update(&mut self, rect: Rect) {
        let x = rect.x as f32 + rect.width as f32 / 2.0;
        let y = rect.y as f32 + rect.height as f32 / 2.0;
        let area = (rect.width * rect.height) as f32;
        let ratio = rect.width as f32 / rect.height as f32;

        let z = MeasVec::new(x, y, area, ratio);

        let mut height = MeasMat::zeros();
        for i in 0..4 {
            height[(i, i)] = 1.0;
        }

        let mut r = SMatrix::<f32, 4, 4>::identity();
        for i in 0..4 {
            r[(i, i)] = if i < 2 { 1.0 } else { 10.0 };
        }

        // Standard Kalman Equations:
        //
        // $$ s = h * p * h^t + r $$
        //
        // Innovation Covariance: Predicted Uncertainty + Sensor Noise.
        let p_times_h_to_the_t = self.covariance * height.transpose();
        let s = (height * p_times_h_to_the_t) + r;

        if let Some(s_inv) = s.try_inverse() {
            // Kalman Gain:
            //
            // $$ k = p * h^t * s^-1 $$
            //
            // If Sensor Noise ($r$) is low, $k$ is high. If Prediction
            // Uncertainty ($p$) is low, $k$ is low. We either trust the
            // measurment or the prediction.
            let k = p_times_h_to_the_t * s_inv;
            let y_residual = z - (height * self.state);
            self.state += k * y_residual;

            // Correct the Uncertainty: $ p = (i - k * h) * p $.
            //
            // Uncertainty shrinks after an update because we confirmed the location.
            self.covariance =
                (StateMat::identity() - (k * height)) * self.covariance;
        }
    }

    fn state_to_rect(&self) -> Rect {
        let x = self.state[0];
        let y = self.state[1];
        let area = self.state[2];
        let ratio = self.state[3];

        let width = (area * ratio).sqrt();
        let height = area / width;

        if width < 1.0 || height < 1.0 {
            return Rect::default();
        }

        Rect::new(
            (x - width / 2.0) as i32,
            (y - height / 2.0) as i32,
            width as i32,
            height as i32,
        )
    }
}
