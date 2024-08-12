import numpy as np


class Simulator:
    def __init__(
        self,
        state_space,
        controller,
        observer,
        initial_state,
        desired_state,
        simulation_time=10,
        dt=0.01,
        dead_time=0.05,
        add_measurement_noise=False,
        use_estimates=True,
        use_quantize=False,
        encoder_resolution=72,
    ):
        self.state_space = state_space
        self.controller = controller
        self.observer = observer
        self.initial_state = initial_state
        self.desired_state = desired_state
        self.use_estimates = use_estimates
        self.dt = dt
        self.dead_time = dead_time
        self.R_obs = np.eye(state_space.C.shape[0]) * 0.001
        self.add_measurement_noise = add_measurement_noise
        self.encoder_resolution = encoder_resolution
        self.u_buffer = []
        self.state = initial_state
        self.states = []
        self.estimated_states = []
        self.observed_states = []
        self.control_inputs = []
        self.delayed_inputs = []
        self.time = np.arange(0, simulation_time, dt)
        self.integral_errors = 0
        self.use_quantize = use_quantize

        self.success_time = None
        self.diff_history = []

    def update_and_get_delayed_input(self, t, u):
        self.u_buffer.append((t, u))

        if t < self.dead_time:
            u_delayed = np.array([0])
        elif len(self.u_buffer) >= 1:
            u_delayed = self.u_buffer[0][1]
        else:
            u_delayed = np.array([0])

        while len(self.u_buffer) > 0 and self.u_buffer[0][0] < t - self.dead_time:
            self.u_buffer.pop(0)

        return u_delayed

    def quantize(self, value):
        quantized_value = np.round(value * self.encoder_resolution / (2 * np.pi)) * (
            2 * np.pi / self.encoder_resolution
        )
        return quantized_value

    def run(self):
        for ti in self.time:
            if self.use_estimates:
                current_state = self.observer.get_state_estimate()
            else:
                current_state = self.state

            u = self.controller.control(current_state, self.desired_state, ti)
            u_clip = np.clip(u, -500, 500)
            u_delayed = (
                self.update_and_get_delayed_input(ti, u_clip) if self.dead_time > 0.0 else u_clip
            )
            self.control_inputs.append(u)
            self.delayed_inputs.append(u_delayed)

            self.state = self.runge_kutta_step(
                self.state_space.dynamics.update_state,
                self.state,
                ti,
                self.dt,
                u_delayed,
            )
            self.state = self.normalize_state(self.state)
            self.states.append(self.state)

            diff = np.abs(self.state[0] - self.desired_state[0]) + np.abs(
                self.state[2] - self.desired_state[2]
            )
            self.diff_history.append(diff)
            success_length = int(1 / self.dt)
            is_success = False
            if len(self.diff_history) > success_length:
                is_success = np.all(np.array(self.diff_history[-success_length:]) < np.deg2rad(5))
                if is_success and self.success_time is None:
                    self.success_time = ti

            self.observer.predict(u_delayed)
            if self.state_space.C.shape[0] == 4:
                if self.add_measurement_noise:
                    y_k = self.state + np.random.multivariate_normal(
                        np.zeros(self.state.shape), self.R_obs
                    )
                else:
                    y_k = self.state
            elif self.state_space.C.shape[0] == 2:
                if self.add_measurement_noise:
                    y_k = self.state[[0, 2]] + np.random.multivariate_normal(
                        np.zeros(2), self.R_obs
                    )
                else:
                    y_k = self.state[[0, 2]]
            if self.use_quantize:
                y_k = self.quantize(y_k)
            self.observed_states.append(y_k)
            self.observer.update(y_k)
            self.estimated_states.append(self.observer.get_state_estimate())

        return (
            np.array(self.states),
            np.array(self.estimated_states),
            np.array(self.observed_states),
            np.array(self.control_inputs),
            np.array(self.delayed_inputs),
            np.array(self.diff_history),
            self.success_time,
        )

    def normalize_state(self, state):
        state[0] = (state[0] + np.pi) % (2 * np.pi) - np.pi
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        return state

    def runge_kutta_step(self, func, x, t, dt, u):
        k1 = func(x, t, u)
        k2 = func(x + 0.5 * dt * k1, t + 0.5 * dt, u)
        k3 = func(x + 0.5 * dt * k2, t + 0.5 * dt, u)
        k4 = func(x + dt * k3, t + dt, u)
        y = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y
