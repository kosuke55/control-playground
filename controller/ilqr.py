import numpy as np
from ilqr import iLQR
from ilqr.containers import Cost, Dynamics

from controller.base import Controller


class ILQRControllerNumba(Controller):
    def __init__(
        self,
        dynamics,
        desired_state,
        Q,
        R,
        QT,
        N,
        dt,
        horizon_dt,
        maxiters=50,
        early_stop=True,
    ):
        super().__init__(dt)
        self.maxiters = maxiters
        self.early_stop = early_stop

        c1 = dynamics.c1
        c2 = dynamics.c2
        alpha1 = dynamics.alpha1
        alpha2 = dynamics.alpha2
        alpha3 = dynamics.alpha3
        alpha4 = dynamics.alpha4
        alpha5 = dynamics.alpha5

        def f(x, u):
            theta1, theta1_dot, theta2, theta2_dot = x
            theta12 = theta1 - theta2
            cos_theta12 = np.cos(theta12)
            sin_theta12 = np.sin(theta12)
            denominator = alpha1 * alpha2 - alpha3**2 * cos_theta12**2
            if denominator == 0:
                raise ValueError("Denominator is zero, causing division by zero")
            theta1_ddot = (
                -alpha2 * alpha3 * sin_theta12 * theta2_dot**2
                + alpha2 * alpha4 * np.sin(theta1)
                - alpha2 * c1 * theta1_dot
                - alpha2 * c2 * theta1_dot
                + alpha2 * c2 * theta2_dot
                + alpha2 * u[0]
                - alpha3**2 * sin_theta12 * cos_theta12 * theta1_dot**2
                - alpha3 * alpha5 * np.sin(theta2) * cos_theta12
                - alpha3 * c2 * cos_theta12 * theta1_dot
                + alpha3 * c2 * cos_theta12 * theta2_dot
            ) / denominator
            theta2_ddot = (
                alpha1 * alpha3 * sin_theta12 * theta1_dot**2
                + alpha1 * alpha5 * np.sin(theta2)
                + alpha1 * c2 * theta1_dot
                - alpha1 * c2 * theta2_dot
                + alpha3**2 * sin_theta12 * cos_theta12 * theta2_dot**2
                - alpha3 * alpha4 * np.sin(theta1) * cos_theta12
                + alpha3 * c1 * cos_theta12 * theta1_dot
                + alpha3 * c2 * cos_theta12 * theta1_dot
                - alpha3 * c2 * cos_theta12 * theta2_dot
                - alpha3 * u[0] * cos_theta12
            ) / denominator
            dxdt = np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot], dtype=np.float64)
            return x + dxdt * horizon_dt

        cost = Cost.QR(Q, R, QT, desired_state)
        discrete_dynamics = Dynamics.Discrete(f)

        self.iLQR = iLQR(discrete_dynamics, cost)
        self.u = np.zeros((N, 1))

    def control(self, state, desired_state, t):
        if self.last_update_time is None or np.round(t - self.last_update_time, 2) >= self.dt:
            self.last_update_time = t

            xs, us, cost_trace = self.iLQR.fit(
                state, self.u, maxiters=self.maxiters, early_stop=self.early_stop
            )
            self.u = us

        return self.u[0]
