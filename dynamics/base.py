class Dynamics:
    def create_state_space(self):
        raise NotImplementedError

    def update_state(self, state, t, u):
        raise NotImplementedError
