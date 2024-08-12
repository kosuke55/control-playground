import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P = np.eye(A.shape[0])
        self.state_estimate = np.zeros((A.shape[0],))

    def predict(self, u):
        self.state_estimate = self.A @ self.state_estimate + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        Kf = self.P @ self.C.T @ inv(self.C @ self.P @ self.C.T + self.R)
        self.state_estimate += Kf @ (y - self.C @ self.state_estimate)
        self.P = (np.eye(len(self.P)) - Kf @ self.C) @ self.P

    def get_state_estimate(self):
        return self.state_estimate
