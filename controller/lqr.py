import control
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are

from controller.base import Controller
from utils.math import sign


class LQRController(Controller):
    def __init__(self, A, B, Q, R, dt):
        super().__init__(dt)
        self.K = self._calculate_lqr_gain(A, B, Q, R)

        self.use_friction_compensation = False
        self.f1 = 0.0
        self.f2 = 0.0
        self.friction_compensation = 0
        self._friction_compensations = []

    def _calculate_lqr_gain(self, A, B, Q, R):
        X = solve_continuous_are(A, B, Q, R)
        K = inv(R) @ B.T @ X
        return K

    def control(self, state, desired_state, t):
        if self.last_update_time is None or np.round(t - self.last_update_time, 2) >= self.dt:
            self.last_update_time = t
            self.u = -self.K @ (state - desired_state)

            if self.use_friction_compensation:
                self.friction_compensation = (
                    self.f1 * sign(state[1]) + self.f2 * sign(state[1] + state[3])
                ) * 1
                self.u += self.friction_compensation

        if self.use_friction_compensation:
            self._friction_compensations.append(self.friction_compensation)

        return self.u

    def get_friction_compensations(self):
        return self._friction_compensations


class LQRControllerWithIntegral(Controller):
    def __init__(self, A, B, C, Q, R, dt):
        super().__init__(dt)
        self.A_aug, self.B_aug, self.C_aug = self._augment_matrices(A, B, C)
        self.K = self._calculate_lqr_gain(self.A_aug, self.B_aug, Q, R)
        # self.integral_error = np.zeros(C.shape[0])
        self.integral_error = np.zeros(1)

        self.use_friction_compensation = True
        self.f1 = 0.3
        self.f2 = 0.3
        self.friction_compensation = 0
        self._friction_compensations = []

    def _augment_matrices(self, A, B, C):
        n = A.shape[0]
        p = 1  # θ1, θ2のみの誤差を状態量に含めると可制御ではなくなるため、θ1のみの誤差を状態量に含める
        A_aug = np.zeros((n + p, n + p))
        A_aug[:n, :n] = A
        A_aug[:n, n:] = np.zeros((n, p))
        A_aug[n:, :n] = -np.array([[1, 0, 0, 0]])

        B_aug = np.zeros((n + p, B.shape[1]))
        B_aug[:n, :] = B

        Wc = control.ctrb(A_aug, B_aug)
        print(f"Wc: {Wc}")
        if np.linalg.matrix_rank(Wc) != n + p:
            print("System not Controllability\n")
        else:
            print("System Controllability\n")

        C_aug = np.zeros((C.shape[0], C.shape[1] + p))
        C_aug[:, : C.shape[1]] = C

        return A_aug, B_aug, C_aug

    def _calculate_lqr_gain(self, A, B, Q, R):
        X = solve_continuous_are(A, B, Q, R)
        K = inv(R) @ B.T @ X
        return K

    def control(self, state, desired_state, t):
        if self.last_update_time is None or np.round(t - self.last_update_time, 2) >= self.dt:
            self.last_update_time = t
            error = desired_state[0] - state[0]  # θ1の誤差のみを積分
            self.integral_error += error * self.dt
            augmented_state = np.hstack((state - desired_state, self.integral_error))

            self.u = -self.K @ augmented_state
            if self.use_friction_compensation:
                self.friction_compensation = (
                    self.f1 * sign(state[1]) + self.f2 * sign(state[1] + state[3])
                ) * 1
                self.u += self.friction_compensation

        if self.use_friction_compensation:
            self._friction_compensations.append(self.friction_compensation)

        return self.u

    def get_friction_compensations(self):
        return self._friction_compensations
