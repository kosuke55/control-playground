import numpy as np


class Controller:
    def __init__(self, dt):
        self.dt = dt
        self.last_update_time = None

    def control(self, state, desired_state, t):
        raise NotImplementedError

    def get_friction_compensations(self):
        return np.array([])
