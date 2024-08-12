import numpy as np

from controller.base import Controller
from utils.math import sign


class PIDController(Controller):
    def __init__(self, Kp, Ki, Kd, dt):
        super().__init__(dt)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0

        self.use_friction_compensation = True
        self.f1 = 0.3
        self.f2 = 0.3
        self.friction_compensation = 0
        self._friction_compensations = []

    def control(self, state, desired_state, t):
        if self.last_update_time is None or np.round(t - self.last_update_time, 2) >= self.dt:
            self.last_update_time = t
            error = desired_state - state
            self.integral += error * self.dt
            derivative = (error - self.previous_error) / self.dt
            self.u = np.array(
                [
                    np.dot(self.Kp, error)
                    + np.dot(self.Ki, self.integral)
                    + np.dot(self.Kd, derivative)
                ]
            )
            if self.use_friction_compensation:
                self.friction_compensation = (
                    self.f1 * sign(state[1]) + self.f2 * sign(state[1] + state[3])
                ) * 1
                self.u += self.friction_compensation

            self.previous_error = error

        if self.use_friction_compensation:
            self._friction_compensations.append(self.friction_compensation)

        return self.u

    def get_friction_compensations(self):
        return self._friction_compensations
