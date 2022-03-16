import numpy as np

class Quadrotor:
    def __init__(self, time_step=0.3, g=9.8, random_state=None,
                 drag_alpha_x_bounds = (0.3, 0.7),
                 drag_alpha_y_bounds = (0.3, 0.7),
                 drag_alpha_x_nom = 0.5,
                 drag_alpha_y_nom = 0.5,
                 disturbance_boundary_sample_rate=0.
                 ):
        '''
        s = [x ẋ y ẏ]ᵀ
        s⁺ = As+Bu+Bw
        '''
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state=random_state
        self.n_x = 4
        self.n_u = 2

        self.time_step = time_step
        self.g = g
        self.A = np.zeros((self.n_x, self.n_x))
        self.A[0:2, 2:] = np.eye(2)
        self.B = np.zeros((self.n_u, self.n_x))
        self.B[0, 2] = self.g
        self.B[1, 3] = -self.g
        self.B = np.transpose(self.B)
        self.I4 = np.eye(self.n_x)

        # drag coefficient
        self.drag_alpha_x_bounds = drag_alpha_x_bounds
        self.drag_alpha_y_bounds = drag_alpha_y_bounds
        self.drag_alpha_x_nom = drag_alpha_x_nom
        self.drag_alpha_y_nom = drag_alpha_y_nom
        # memoized drag coefficient corresponding to the particle index
        self.drag_alpha_x_samples = dict()
        self.drag_alpha_y_samples = dict()
        self.disturbance_boundary_sample_rate = disturbance_boundary_sample_rate
        # lipschitz constants for nonlinear disturbance
        max_velocity = 0.5  # [m/s]
        self.d_lip_constants = np.array([0., 0.,
                                         2. *
                                         self.drag_alpha_x_bounds[1] *
                                         max_velocity,
                                         2. *
                                         self.drag_alpha_y_bounds[1] *
                                         max_velocity
                                         ])
        # feedback controller
        self.K = -0.1*np.array([[1., 0., 0.1, 0.],
                                [0., -1., 0.0, -0.1]])

    def simulate_forward(self, x, local_controller, *args, **kwargs):
        """
        Implement forward dynamics through self.f_dt
        This is the function called by RandUP-RRT directly
        :param x: state to start from
        :param local_controller: function that returns the control input
        :return: (resulting_state, resulting_mode, has_collision)
        """
        # The drag coefficient is associated with the particular particle
        particle_index = kwargs["particle_index"]

        # TODO make function dependent on dt?
        x_next = self.f_dt(x, local_controller(x), particle_index=particle_index)
        return x_next, kwargs["current_mode"], False

    # ----------------------------------------------
    # Uncertainty propagation using lipschitz method
    def A_dt(self, dt):
        return (self.I4+(self.A*dt)/2.0)**2

    def B_dt(self, dt):
        return (dt*self.I4+self.A*(dt/2.0)**2)@self.B

    def d_dt(self, x_k, u_k, dt, drag_alpha_x=None, drag_alpha_y=None):
        """
        Drag term.
        :param x_k:
        :param u_k:
        :param dt:
        :param drag_alpha_x: Use self.drag_alpha_x_nom if set to None
        :param drag_alpha_y: Use self.drag_alpha_y_nom if set to None
        :return:
        """
        # predicts disturbance (wind drag)
        # - x_k : [n_x]    current state
        # - u_k : [n_u]    current control input
        # - dt  : [   ]    discretization time
        d = np.zeros_like(x_k)

        # TODO REPLACE - ADD ALPHA PARAMETERS AS ARGUMENT
        if drag_alpha_x is None:
            drag_alpha_x = self.drag_alpha_x_nom
        if drag_alpha_y is None:
            drag_alpha_y = self.drag_alpha_y_nom
        d[2] = -drag_alpha_x * np.sign(x_k[2]) * (x_k[2] ** 2)
        d[3] = -drag_alpha_y * np.sign(x_k[3]) * (x_k[3] ** 2)
        d = dt * d
        return d

    def f_dt(self, x_k, u_k, dt=None, particle_index=None):
        if dt is None:
            dt = self.time_step
        # predicts the next state given
        # - x_k : [n_x]    current state
        # - u_k : [n_u]    current control input
        # - dt  : [   ]    discretization time
        if particle_index is not None:
            if particle_index not in self.drag_alpha_x_samples.keys():
                # Sample randomly
                if self.random_state.uniform(0,1)<self.disturbance_boundary_sample_rate:
                    self.drag_alpha_x_samples[particle_index] = self.random_state.choice(self.drag_alpha_x_bounds, 1)[0]
                else:
                    self.drag_alpha_x_samples[particle_index] = self.random_state.uniform(
                                                                    self.drag_alpha_x_bounds[0],
                                                                    self.drag_alpha_x_bounds[1])
            if particle_index not in self.drag_alpha_y_samples.keys():
                # sample randomly
                if self.random_state.uniform(0,1)<self.disturbance_boundary_sample_rate:
                    self.drag_alpha_y_samples[particle_index] = self.random_state.choice(self.drag_alpha_y_bounds, 1)[0]
                else:
                    self.drag_alpha_y_samples[particle_index] = self.random_state.uniform(
                                                                    self.drag_alpha_y_bounds[0],
                                                                    self.drag_alpha_y_bounds[1])
            drag_alpha_x = self.drag_alpha_x_samples[particle_index]
            drag_alpha_y = self.drag_alpha_y_samples[particle_index]
        else:
            drag_alpha_x = self.drag_alpha_x_nom
            drag_alpha_y = self.drag_alpha_y_nom
        x_next = self.A_dt(dt)@x_k+self.B_dt(dt)@u_k + \
            self.d_dt(x_k, u_k, dt, drag_alpha_x, drag_alpha_y)
        return x_next
