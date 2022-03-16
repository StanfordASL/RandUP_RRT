class ForwardDynamicsSimulator:
    def __init__(self):
        raise NotImplementedError

    def simulate_forward(self, starting_state, local_controller,
                         step_size=None, number_of_steps=None,
                         desired_mode=None):
        raise NotImplementedError
