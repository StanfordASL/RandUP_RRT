from planning.contact_modes import HybridIntegratorModes
import planning.randup_rrt as randup_rrt
import planning.param as param
import hybrid_integrator.hybrid_integrator as hybrid_integrator
import hybrid_integrator.visualization.plot_hybrid_integrator as plt_rrt
import utils.umath as umath

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
import os
from collections import deque
from matplotlib.patches import Circle, Rectangle

# In this parameterization, the action is the coefficient of the
# friction cone basis ranging from 0~max
distance_scaling_array = np.array([1., 1., 1., 0.1])
goal_distance_scaling_array = np.array([1., 0., 1., 0.])


def reached_goal(state, mode):
    dist = np.linalg.norm(goal_distance_scaling_array * (state - goal_state),
                          axis=1)
    return np.all(dist < goal_threshold), np.max(dist)

def plan_hybrid_integrator(root_state, root_mode, goal_state, hi_system, world,
                           max_node_count=100,
                           randup_state_count=1,
                           goal_bias=0.02,
                           action_radius=1.,
                           p_gain=100.,
                           d_gain=2.,
                           goal_threshold=0.5,
                           number_of_steps=20,
                           flight_mode_sample_probability=0.7,
                           vy_max=11.,
                           random_state=None):
    def local_controller_creator(sample_node, sample_action,
                                 sample_next_mode, forward_dynamics_sim):
        def local_controller(current_state, current_mode):
            action = np.zeros(2)
            action[0] = p_gain * (sample_action[0] - current_state[0]) \
                + d_gain * (sample_action[2] - current_state[2])
            action[1] = sample_action[2]
            desired_next_mode = HybridIntegratorModes.CONTACT
            # if current_mode == HybridIntegratorModes.CONTACT:
            # if sample_node.contact_mode== HybridIntegratorModes.CONTACT:
            #     # if sample_node.parent.contact_mode == HybridIntegratorModes.CONTACT:
            #         desired_next_mode = HybridIntegratorModes.FLIGHT
            return sample_next_mode, action
        return local_controller

    sample_state_highs = np.array([world.world_bounds[1, 0],
                                   hi_system.xdot_max,
                                   world.world_bounds[1, 1],
                                   hi_system.ydot_max])
    sample_state_lows = -np.copy(sample_state_highs)
    sample_state_lows[[0, 2]] = np.copy(world.world_bounds[0, :])
    sample_state_lows[1] = -hi_system.xdot_max
    sample_state_lows[3] = -0.2*hi_system.ydot_max

    sampled_actions = []
    sampled_states = []

    def action_sampler(sampled_node):
        action_sample = np.zeros(4)
        d = np.array([np.sqrt(random_state.uniform(0., action_radius**2)),
                      0.])
        action_vec = umath.get_R_AB(random_state.uniform(0, 2 * np.pi)) @ d
        action_sample[0] = action_vec[0]+sampled_node.nominal_state[0]
        action_sample[2] = np.sqrt(random_state.uniform(vy_max**2))
        sampled_actions.append(action_sample)
        return action_sample

    def forward_dynamics_sim(starting_state, local_controller,
                             current_mode, particle_index):
        return hi_system.simulate_forward(
            starting_state, local_controller, number_of_steps=number_of_steps,
            starting_mode=current_mode, particle_index=particle_index,
            collision_type_function=world.get_collision_type_fn()
        )

    def state_sampler():
        if random_state.random(1) < goal_bias:
            sample = goal_state
        else:
            sample = random_state.uniform(
                low=sample_state_lows, high=sample_state_highs)
        sampled_states.append(sample)
        return sample


    def is_safe(states):
        return True

    def mode_sampler(sample_node):
        if random_state.rand() < flight_mode_sample_probability and\
                sample_node.contact_mode == HybridIntegratorModes.CONTACT:
            if sample_node.parent is not None:
                if sample_node.parent.contact_mode == HybridIntegratorModes.CONTACT:
                    return HybridIntegratorModes.FLIGHT
            else:
                return HybridIntegratorModes.FLIGHT
        return HybridIntegratorModes.CONTACT
    rrt = randup_rrt.RandUpRRT(root_state, root_mode,
                               forward_dynamics_sim,
                               randup_state_count,
                               state_sampler, action_sampler,
                               mode_sampler,
                               modes_enum=HybridIntegratorModes,
                               distance_scaling_array=distance_scaling_array,
                               random_state=random_state)
    start_time = time.time()
    rrt.plan(reached_goal, local_controller_creator, is_safe, max_node_count,
             print_interval=100)
    end_time = time.time()
    print("Plan generation time: ", end_time-start_time)
    # sampled_actions = np.asarray(sampled_actions)
    # sample_states = np.asarray(sampled_states)
    # plt.scatter(sampled_actions[:,0], sampled_actions[:,2],color='r', s=0.5)
    # plt.scatter(sample_states[:,0], sample_states[:,2],color='b', s=0.5)
    # plt.show()
    return rrt


def animate_hybrid_integrator_rrt(rrt, actions, local_controllers, number_of_steps,
                             number_of_randup_particles, hybrid_integrator_system,
                             world, folder_name, goal_state, goal_radius):
    # current_time = str(time.time())
    # folder_name = current_time
    folder_name = folder_name.split('.')[0]
    os.mkdir(folder_name)
    states = np.tile(rrt.root_state, (number_of_randup_particles, 1))
    modes = np.tile(rrt.root_mode, (number_of_randup_particles, 1))
    colors = np.random.rand(number_of_randup_particles, 3)
    x_min, x_max = world.world_bounds[0, :]
    y_min, y_max = world.world_bounds[1, :]

    for controller_i, local_controller in enumerate(local_controllers):
        override_control_modes = []
        for step_i in range(number_of_steps):
            # save the image
            image_id = controller_i*number_of_steps+step_i
            fig, ax = plt.subplots()
            # if step_i == 0:
            #     for pi in range(number_of_randup_particles):
            #         override_control_mode, _ = local_controller(states[pi, :],
            #                                                     modes[pi, :])
            #         override_control_modes.append(override_control_mode)
            for pi in range(number_of_randup_particles):
                if step_i == 0:
                    override_control_mode, _ = local_controller(states[pi, :],
                                                                modes[pi, :])
                ans = hybrid_integrator_system.simulate_forward(
                    states[pi, :],
                    local_controller,
                    1,
                    world.get_collision_type_fn(),
                    modes[pi], pi,
                    override_step_i=step_i)
                if ans[2]:
                    warnings.warn('Collision')
                states[pi, :] = ans[0]
                modes[pi] = ans[1]
                if modes[pi] == HybridIntegratorModes.CONTACT:
                    marker = 'o'
                else:
                    marker = 'X'
                ax.scatter(states[pi, 0], states[pi, 2], marker=marker,
                           color=colors[pi])
            world.visualize(ax)
            goal_circle = Circle(goal_state[[0, 2]], goal_radius, color='g',
                                 alpha=0.3)
            ax.add_patch(goal_circle)

            # plot world box
            # # x_bottom
            # ax.plot(np.linspace(x_min, x_max, 10), y_min*np.zeros(10), 'c-')
            # # x_top
            # ax.plot(np.linspace(x_min, x_max, 10), y_max*np.zeros(10), 'c-')
            # # y_bottom
            # ax.plot(x_min*np.zeros(10),np.linspace(y_min, y_max, 10), 'c-')
            # # y_top
            # ax.plot(x_max*np.zeros(10),np.linspace(y_min, y_max, 10), 'c-')

            ax.set_xlim(left=-0.5, right=15.5)
            ax.set_ylim(bottom=-0.5, top=15.5)
            ax.axis('equal')
            plt.savefig(folder_name+f'/{image_id}.png', dpi=300)
            plt.close(fig)


def execute_hybrid_integrator_plan(local_controllers, number_of_steps,
                                starting_state, starting_mode,
                             number_of_randup_particles,
                                   hybrid_integrator_system,
                                   reached_goal):
    states = np.tile(starting_state, (number_of_randup_particles, 1))
    modes = np.tile(starting_mode, (number_of_randup_particles, 1))
    is_complete = [False]*number_of_randup_particles
    successes = 0
    for controller_i, local_controller in enumerate(local_controllers):
        override_control_modes = []
        for step_i in range(number_of_steps):
            # if step_i == 0:
            #     for pi in range(number_of_randup_particles):
            #         override_control_mode, _ = local_controller(states[pi, :],
            #                                                     modes[pi, :])
            #         override_control_modes.append(override_control_mode)
            for pi in range(number_of_randup_particles):
                if is_complete[pi]:
                    continue
                if step_i == 0:
                    override_control_mode, _ = local_controller(states[pi,:],
                                                                modes[pi,:])
                states[pi,:], modes[pi,:], has_collision = \
                    hybrid_integrator_system.simulate_forward(
                    states[pi,:],
                    local_controller,
                    1,
                    world.get_collision_type_fn(),
                    modes[pi,:], pi,
                    override_step_i=step_i)
                if has_collision:
                    warnings.warn(f"Particle {pi} Collision")
                    is_complete[pi] = True
                # Check for goals
                is_at_goal, _ = reached_goal(np.atleast_2d(states[pi,:]),
                                np.atleast_2d(modes[pi,:]))
                if is_at_goal:
                    print(f"Particle {pi} reached goal")
                    is_complete[pi] = True
                    successes += 1
                    continue
        return successes/number_of_randup_particles

def rollout_hybrid_integrator_plan(local_controllers, number_of_steps,
                                starting_state, starting_mode,
                                   particle_index,
                                   hybrid_integrator_system,
                                   reached_goal):
    states = [starting_state]
    modes = [starting_mode]
    for controller_i, local_controller in enumerate(local_controllers):
        override_control_modes = []
        for step_i in range(number_of_steps):
            if step_i == 0:
                override_control_mode, _ = local_controller(np.atleast_2d(states[-1]).T,
                                                            np.array([modes[-1]]))
            next_state, next_mode, has_collision = \
                hybrid_integrator_system.simulate_forward(
                states[-1],
                local_controller,
                1,
                world.get_collision_type_fn(),
                modes[-1],particle_index,
                override_step_i=step_i)
            if has_collision:
                warnings.warn("Collision")
                return states, modes, False
            states.append(next_state)
            modes.append(next_mode)
            # Check for goals
            is_at_goal, dist = reached_goal(states[-1], np.atleast_2d(modes[-1]))
        if is_at_goal:
            print(f"Reached goal")
            return states, modes, True
    print(f"Didn't reach goal")
    return states, modes, False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid integrator demo')
    parser.add_argument('--num_particles', type=int, default=100,
                        required=False,
                        help='Number of RandUP particles. Use 1 for RRT.')
    args = parser.parse_args()
    randup_state_count = args.num_particles
    assert 1<=randup_state_count
    print(f'Using {randup_state_count} particles.')
    seed = 5
    print('Seed', seed)
    random_state = np.random.RandomState(seed)
    max_node_count = 30000
    root_state = np.asarray([1., 0., 0., 0.])
    root_mode = HybridIntegratorModes.CONTACT
    goal_state = np.asarray([2., 0., 4., 0.])
    playback_verification_count = 1000
    time_step = 3e-2
    number_of_steps = 7
    goal_threshold = 0.8
    xdot_max = 10.
    ydot_max = 20.
    vy_max = 9  # 8.1
    max_jump_delay = 3
    action_radius = 1.2
    action_scaling_range = (0.96, 1.)
    print(f'Parameters: \nmax node {max_node_count}'
          f'\nroot state {root_state} \ngoal state {goal_state}'
          f'\nrandup state count {randup_state_count}'
          f'\naction scaling range {action_scaling_range}'
          f'\nmax jump delay {max_jump_delay}')

    # # Construct the obstacles
    obstacles_array = np.zeros((5, 2, 4))
    obstacles_array[:, 0, [1, 3]] = -np.inf
    obstacles_array[:, 1, [1, 3]] = np.inf
    obstacles_array[0, [0, 1], 0] = [0, 6]
    obstacles_array[0, [0, 1], 2] = [3., 3.5]
    obstacles_array[1, [0, 1], 0] = [5.2, 8]
    obstacles_array[1, [0, 1], 2] = [0.5, 1.]
    obstacles_array[2, [0, 1], 0] = [8, 13]
    obstacles_array[2, [0, 1], 2] = [1.5, 2.3]
    obstacles_array[3, [0, 1], 0] = [11, 14]
    obstacles_array[3, [0, 1], 2] = [5, 6]

    world_bounds = np.asarray([[0, 0], [15, 15]])
    world = hybrid_integrator.AABBHybridIntegratorObstacleWorld(obstacles_array,
                                                                world_bounds)
    hybrid_integrator_system = hybrid_integrator.HybridIntegrator(
        time_step=time_step,
        random_state=random_state,
        xdot_max=xdot_max,
        ydot_max=ydot_max,
        action_scaling_range=action_scaling_range,
        max_jump_delay=max_jump_delay,
        randup_state_count=randup_state_count,
        gravity=9.8)

    rrt = plan_hybrid_integrator(root_state, root_mode, goal_state,
                                 hybrid_integrator_system,
                                 world,
                                 max_node_count,
                                 action_radius=action_radius,
                                 number_of_steps=number_of_steps,
                                 randup_state_count=randup_state_count,
                                 goal_threshold=goal_threshold,
                                 vy_max=vy_max,
                                 random_state=random_state)
    print(f'S2N map length = {len(rrt.state_to_node_map)}')
    node_count = 0
    for k in rrt.state_to_node_map.keys():
        node_count += len(rrt.state_to_node_map[k])


    # Validate on 1000 particles
    def reached_goal_val(state, mode):
        dist = np.linalg.norm(
            goal_distance_scaling_array * (state - goal_state))
        return np.all(dist < goal_threshold), np.max(dist)

    hybrid_integrator_system_val = hybrid_integrator.HybridIntegrator(
        time_step=time_step,
        random_state=random_state,
        xdot_max=xdot_max,
        ydot_max=ydot_max,
        action_scaling_range=action_scaling_range,
        max_jump_delay=max_jump_delay,
        randup_state_count=playback_verification_count,
        gravity=9.8)
    actions, local_controllers, expected_modes = rrt.backtrack_actions(
        rrt.goal_node)
    # Plotting
    fig, ax = plt_rrt.visualize_hybrid_integrator_rrt(
        rrt, goal_state, show_all_nodes=False,  # node_count < 1500,
        world=world, goal_radius=goal_threshold)
    # rollout 50 trajectories
    safe_count = 0
    execution_count = 50
    for i in range(execution_count):
        states, modes, is_safe = rollout_hybrid_integrator_plan(
            local_controllers,
            number_of_steps,
            rrt.root_state, rrt.root_mode,
            i,
            hybrid_integrator_system_val,
            reached_goal_val)
        xy_states = np.array(states)[:, [0, 2]]
        # Plotting
        # plt.show()
        if is_safe:
            safe_count += 1
            color = 'teal'
        else:
            color = 'magenta'
        ax.plot(xy_states[:, 0], xy_states[:, 1], color=color, ls='-',
                alpha=0.035)
    print(f"Safe executions: {safe_count}/{execution_count}")
    filename = f'{time.strftime("%Y%m%d-%H%M%S")}_' \
               f'randup_{randup_state_count}_nodes_{node_count}_seed_{seed}_delay_{max_jump_delay}.png'
    plt.savefig(filename, dpi=300, bbox_inches="tight")