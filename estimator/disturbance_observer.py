import numpy as np


class DisturbanceObserver:
    def __init__(self, A, B, C, L):
        self.A = A
        self.B = B
        self.C = C
        self.L = L
        self.disturbance_estimate = np.zeros((A.shape[0],))
        self.state_estimate = np.zeros((A.shape[0],))

    def predict(self, u):
        self.state_estimate = self.A @ self.state_estimate + self.B @ u

    def update(self, y):
        y_hat = self.C @ self.state_estimate
        self.disturbance_estimate = self.L @ (y - y_hat)
        self.state_estimate += self.disturbance_estimate

    def get_state_estimate(self):
        return self.state_estimate

    def get_disturbance_estimate(self):
        return self.disturbance_estimate
