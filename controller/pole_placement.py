import control
import numpy as np
from scipy.signal import place_poles

from controller.base import Controller
from utils.math import sign


class PolePlacementController(Controller):
    def __init__(self, A, B, desired_poles, dt):
        super().__init__(dt)
        self.K = self._calculate_pole_placement_gain(A, B, desired_poles)
        print(f"K: {self.K}")

        self.use_friction_compensation = False
        self.f1 = 0.3
        self.f2 = 0.3
        self.friction_compensation = 0
        self._friction_compensations = []

    def _calculate_pole_placement_gain(self, A, B, desired_poles):
        # Calculate the feedback gain using pole placement
        K = control.place(A, B, desired_poles)
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
