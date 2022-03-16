import utils.umath as umath
import planning.contact_modes as contact_modes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from enum import Enum


class HybridIntegrator:
    def __init__(self, time_step=1e-3, random_state=None,
                 xdot_max=5., ydot_max=15.,
                 action_scaling_range=(0.85, 1.),
                 max_jump_delay=3., gravity=9.8,
                 randup_state_count=20,
                 action_in_flight_multiplier=1.):
        '''
        s = [x ẋ y ẏ]ᵀ
        s⁺ = As+Bu+Bw
        '''
        self.time_step = time_step
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state
        self.gravity = gravity

        self.randup_state_count = randup_state_count
        self.xdot_max = xdot_max
        self.ydot_max = ydot_max
        # Generate the random x_dot_max
        self.action_scaling_range = action_scaling_range
        self.action_scaling = self.random_state.uniform(self.action_scaling_range[0],
                                                        self.action_scaling_range[1],
                                                        self.randup_state_count)
        self.max_jump_delay = max_jump_delay
        self.action_in_flight_multiplier = action_in_flight_multiplier

    def simulate_forward(self, starting_state, local_controller,
                         number_of_steps=100,
                         collision_type_function=None,
                         starting_mode=None,
                         particle_index=None,
                         override_control_mode=None,
                         override_step_i=-1):
        assert collision_type_function is not None
        assert starting_mode is not None
        assert particle_index is not None
        state = np.copy(starting_state)
        random_jump_delay = self.random_state.randint(low=-1,
                                                      high=self.max_jump_delay,
                                                      size=1)
        # step = 0
        # while step < number_of_steps or (not self.allow_action_in_flight and
        #                    current_mode==contact_modes.HybridIntegratorModes.FLIGHT):
        #     step += 1
        current_mode = starting_mode
        for step in range(number_of_steps):
            if step == 0:
                control_mode, control_action = local_controller(state,
                                                                current_mode)
            else:
                _, control_action = local_controller(state,
                                                     current_mode)
            if override_control_mode is not None:
                control_mode = override_control_mode
            previous_state = np.copy(state)
            # x simulation is independent of flight
            state[0] += state[1] * self.time_step
            if current_mode == contact_modes.HybridIntegratorModes.FLIGHT:
                state[1] += control_action[0] * self.time_step * \
                    self.action_scaling[particle_index] * \
                    self.action_in_flight_multiplier
            else:
                state[1] += control_action[0] * self.time_step * \
                    self.action_scaling[particle_index]
            # Clip xdot
            state[1] = min(max(state[1], -self.xdot_max), self.xdot_max)
            if current_mode == contact_modes.HybridIntegratorModes.CONTACT:
                if control_mode == contact_modes.HybridIntegratorModes.FLIGHT:
                    if override_step_i > random_jump_delay or \
                            step > random_jump_delay:
                        # launch
                        state[3] = control_action[1] * \
                            self.action_scaling[particle_index]
                        current_mode = contact_modes.HybridIntegratorModes.FLIGHT
            elif current_mode == contact_modes.HybridIntegratorModes.FLIGHT:
                # y action cannot be affected after launch
                state[2] += state[3]*self.time_step
                state[3] -= self.gravity * self.time_step
                state[3] = min(max(state[3], -self.ydot_max), self.ydot_max)
            # collision check if state moved
            if not np.allclose(previous_state[[0, 2]], state[[0, 2]]):
                collision_type, collision_height = collision_type_function(previous_state,
                                                                           state)
                if collision_type == HybridIntegratorCollisionType.NO_COLLISION:
                    # No contact
                    current_mode = contact_modes.HybridIntegratorModes.FLIGHT
                elif collision_type == HybridIntegratorCollisionType.TOP_COLLISION:
                    # contact with the top of something
                    # set y velocity to zero
                    state[2] = collision_height
                    if state[3] > 0:
                        current_mode = contact_modes.HybridIntegratorModes.FLIGHT
                    else:
                        current_mode = contact_modes.HybridIntegratorModes.CONTACT
                    state[3] = max(0., state[3])
                elif collision_type == HybridIntegratorCollisionType.OTHER_COLLISION:
                    # contact with the side or bottom of something
                    return None, None, True
                else:
                    raise ValueError
        return state, current_mode, False


class HybridIntegratorCollisionType(Enum):
    NO_COLLISION = 0
    TOP_COLLISION = 1
    OTHER_COLLISION = 2


class AABBHybridIntegratorObstacleWorld:
    def __init__(self, obstacle_list, world_bounds):
        '''

        :param obstacle_list:
        :param world_bounds: [[xmin, ymin], [xmax, ymax]]
        '''
        self.obstacle_aabbs = np.asarray(obstacle_list)
        self.world_bounds = world_bounds
        assert self.obstacle_aabbs.shape[1:] == (2, 4)

    def get_collision_type_fn(self):
        def collision_type_fn(previous_state, current_state):
            if current_state[0] <= self.world_bounds[0, 0] or \
                    current_state[0] >= self.world_bounds[1, 0] or \
                    current_state[2] >= self.world_bounds[1, 1]:
                return HybridIntegratorCollisionType.OTHER_COLLISION, None
            collision_obstacle_indices = np.where(np.logical_and(
                np.all(current_state >= self.obstacle_aabbs[:, 0, :], axis=1),
                np.all(current_state <= self.obstacle_aabbs[:, 1, :], axis=1)))[0]
            if len(collision_obstacle_indices) == 0:
                # check if collide with floor
                if current_state[2] <= self.world_bounds[0, 1]:
                    return HybridIntegratorCollisionType.TOP_COLLISION, \
                           self.world_bounds[0, 1]
                return HybridIntegratorCollisionType.NO_COLLISION, None

            X = np.zeros((2, 2))
            X[0, :] = previous_state[[0, 2]]
            X[1, :] = current_state[[0, 2]]
            ray_coeff = np.linalg.inv(X)@np.ones(2)  # ax+by=1
            assert(len(collision_obstacle_indices) == 1)
            obstacle_index = collision_obstacle_indices[0]
            # Previous state is below top border
            if previous_state[2] < self.obstacle_aabbs[obstacle_index, 1, 2]:
                return HybridIntegratorCollisionType.OTHER_COLLISION, None
            if previous_state[3] > 0:
                return HybridIntegratorCollisionType.OTHER_COLLISION, None
            # Check the top corners
            # As long as g > xddot max, the trajectory is concave and this
            # check is always conservative
            if ray_coeff[0] == 0:
                # sliding on top of the obstacle
                return HybridIntegratorCollisionType.TOP_COLLISION, \
                       self.obstacle_aabbs[obstacle_index, 1, 2]
            x_intercept = (1.-ray_coeff[1] *
                           self.obstacle_aabbs[obstacle_index, 1, 2]) /\
                ray_coeff[0]
            if x_intercept <= self.obstacle_aabbs[obstacle_index, 0, 0] or \
                    x_intercept >= self.obstacle_aabbs[obstacle_index, 1, 0]:
                return HybridIntegratorCollisionType.OTHER_COLLISION, None
            # assume there are no overlapping obstacles or multiple collisions
            return HybridIntegratorCollisionType.TOP_COLLISION, \
                   self.obstacle_aabbs[obstacle_index, 1, 2]
        return collision_type_fn

    def visualize(self, ax, indices=None, alpha=0.5):
        if indices is None:
            indices = [0, 2]
        for obstacle in self.obstacle_aabbs:
            aabb = Rectangle(obstacle[0, indices],
                             *(obstacle[1, indices]-obstacle[0, indices]),
                             color='gray', alpha=alpha)
            ax.add_patch(aabb)
        # Visualize ground
        aabb = Rectangle((-5,-5),
                         25,5,
                         color='gray', alpha=alpha)
        ax.add_patch(aabb)