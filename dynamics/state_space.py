class StateSpace:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.A, self.B, self.C, self.D = self.dynamics.create_state_space()
