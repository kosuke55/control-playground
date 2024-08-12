import casadi as ca
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from controller.base import Controller


class MPCController(Controller):
    def __init__(self, A, B, Q, R, N, dt, horizon_dt):
        super().__init__(dt)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N  # Prediction horizon
        self.dt = dt  # Control period
        self.horizon_dt = horizon_dt  # Horizon period
        self.nx = A.shape[0]
        self.nu = B.shape[1]

        # Discretize system with horizon_dt for prediction
        self.Ad = np.eye(A.shape[0]) + A * horizon_dt
        self.Bd = B * horizon_dt

        # Define optimization variables
        self.cp_U = cp.Variable((self.N, self.nu))

    def predict_state(self, x, u):
        return self.Ad @ x + self.Bd @ u

    def control(self, state, desired_state, t):
        if self.last_update_time is None or np.round(t - self.last_update_time, 2) >= self.dt:
            self.last_update_time = t
            x0 = state
            ref = desired_state

            # Define the optimization problem
            cost = 0
            constraints = []
            x = x0

            for i in range(self.N):
                u = self.cp_U[i, :]
                x = self.predict_state(x, u)
                cost += cp.quad_form(x - ref, self.Q) + cp.quad_form(u, self.R)

            # Constraints (example: input constraints)
            constraints += [self.cp_U <= 100.0, self.cp_U >= -100.0]

            # Solve the optimization problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.OSQP)

            self.u = self.cp_U.value[0, :]

        return self.u


class NonlinearMPCControllerCasADi(Controller):
    def __init__(self, dynamics, A, B, Q, R, N, dt, horizon_dt, integration_method="rk4"):
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        self.N = N  # Prediction horizon
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.integration_method = integration_method
        self.prev_u = np.zeros(self.nu)
        self.horizon_dt = horizon_dt
        self.dt = dt
        self.last_update_time = None
        self.u = np.zeros(self.nu)

        self.alpha1 = dynamics.alpha1
        self.alpha2 = dynamics.alpha2
        self.alpha3 = dynamics.alpha3
        self.alpha4 = dynamics.alpha4
        self.alpha5 = dynamics.alpha5
        self.f1 = dynamics.f1
        self.f2 = dynamics.f2
        self.c1 = dynamics.c1
        self.c2 = dynamics.c2

        # CasADi symbolic variables
        self.state_sym = ca.MX.sym("state", self.nx)
        self.u_sym = ca.MX.sym("u", self.nu)
        self.dxdt_sym = self.f(self.state_sym, 0, self.u_sym)
        self.f_func = ca.Function("f_func", [self.state_sym, self.u_sym], [self.dxdt_sym])

    def predict_state(self, x, u, dt):
        if self.integration_method == "euler":
            return self.euler_step(self.f_func, x, dt, u)
        elif self.integration_method == "rk4":
            return self.runge_kutta_step(self.f_func, x, dt, u)
        else:
            raise ValueError("Invalid integration method. Choose 'euler' or 'rk4'.")

    def euler_step(self, func, x, dt, u):
        return x + dt * func(x, u)

    def runge_kutta_step(self, func, x, dt, u):
        k1 = func(x, u)
        k2 = func(x + 0.5 * dt * k1, u)
        k3 = func(x + 0.5 * dt * k2, u)
        k4 = func(x + dt * k3, u)
        y = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def cost_function(self, U, x0, ref):
        x = x0
        cost = 0
        for i in range(self.N):
            u = U[i * self.nu : (i + 1) * self.nu]
            x = self.predict_state(x, u, self.horizon_dt)
            cost += ca.mtimes([(x - ref).T, self.Q, (x - ref)]) + ca.mtimes([u.T, self.R, u])
        return cost

    def f(self, state, t, u):
        theta1 = state[0]
        theta1_dot = state[1]
        theta2 = state[2]
        theta2_dot = state[3]

        theta12 = theta1 - theta2
        cos_theta12 = ca.cos(theta12)
        sin_theta12 = ca.sin(theta12)
        denominator = self.alpha1 * self.alpha2 - self.alpha3**2 * cos_theta12**2

        theta1_ddot = (
            -self.alpha2 * self.alpha3 * sin_theta12 * theta2_dot**2
            + self.alpha2 * self.alpha4 * ca.sin(theta1)
            - self.alpha2 * self.c1 * theta1_dot
            - self.alpha2 * self.c2 * theta1_dot
            + self.alpha2 * self.c2 * theta2_dot
            + self.alpha2 * u
            - self.alpha3**2 * sin_theta12 * cos_theta12 * theta1_dot**2
            - self.alpha3 * self.alpha5 * ca.sin(theta2) * cos_theta12
            - self.alpha3 * self.c2 * cos_theta12 * theta1_dot
            + self.alpha3 * self.c2 * cos_theta12 * theta2_dot
        ) / denominator

        theta2_ddot = (
            self.alpha1 * self.alpha3 * sin_theta12 * theta1_dot**2
            + self.alpha1 * self.alpha5 * ca.sin(theta2)
            + self.alpha1 * self.c2 * theta1_dot
            - self.alpha1 * self.c2 * theta2_dot
            + self.alpha3**2 * sin_theta12 * cos_theta12 * theta2_dot**2
            - self.alpha3 * self.alpha4 * ca.sin(theta1) * cos_theta12
            + self.alpha3 * self.c1 * cos_theta12 * theta1_dot
            + self.alpha3 * self.c2 * cos_theta12 * theta1_dot
            - self.alpha3 * self.c2 * cos_theta12 * theta2_dot
            - self.alpha3 * u * cos_theta12
        ) / denominator

        dxdt = ca.vertcat(theta1_dot, theta1_ddot, theta2_dot, theta2_ddot)
        return dxdt

    def check_condition_number(self, matrix):
        cond_number = np.linalg.cond(matrix)
        if cond_number > 1e10:
            print(f"Warning: The condition number of the matrix is very high: {cond_number}")
        return cond_number

    def control(self, state, desired_state, t):
        if self.last_update_time is None or np.round(t - self.last_update_time, 2) >= self.dt:
            self.last_update_time = t
            x0 = state
            ref = desired_state

            opti = ca.Opti()

            U = opti.variable(self.N * self.nu)
            x = ca.MX(x0)
            cost = 0
            for i in range(self.N):
                u = U[i * self.nu : (i + 1) * self.nu]
                x = self.predict_state(x, u, self.horizon_dt)
                cost += ca.mtimes([(x - ref).T, self.Q, (x - ref)]) + ca.mtimes([u.T, self.R, u])

            opti.minimize(cost)
            for i in range(self.N * self.nu):
                opti.subject_to(U[i] >= -300)
                opti.subject_to(U[i] <= 300)

            opts = {"ipopt.print_level": 0, "print_time": 0}
            opti.solver("ipopt", opts)

            try:
                sol = opti.solve()
                stats = sol.stats()
                if stats["return_status"] == "Solve_Succeeded":
                    self.u = np.array(sol.value(U)[: self.nu]).flatten()
                    self.last_U = sol.value(U)
                    self.prev_u = self.u
                else:
                    print(f"Optimization failed with status: {stats['return_status']}")
                    self.u = self.prev_u
            except RuntimeError as e:
                print(f"Optimization failed: {e}")
                self.u = self.prev_u

        return self.u
