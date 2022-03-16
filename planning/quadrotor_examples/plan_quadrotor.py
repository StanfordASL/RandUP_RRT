import planning.contact_modes as contact_modes
import planning.randup_rrt as randup_rrt
import quadrotor_planar.visualization.plot_quadrotor_rrt as plt_rrt
import utils.umath as umath
from quadrotor_planar.quadrotor import Quadrotor

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings

goal_threshold = 0.7

# Obstacles setup
two_obstacles = [(np.array([6.5, 2.5]), 2.3),
                  (np.array([3, -2]), 2.3)]

# Normalize
distance_scaling_array = np.array([1., 1., 0.1, 0.1])
scaling_norm = np.linalg.norm(distance_scaling_array)
distance_scaling_array /= np.linalg.norm(distance_scaling_array)
goal_distance_scaling_array = np.array([1., 1., 0., 0.])

def plan_quadrotor(root_state, goal_state,
                   quadrotor,
                   max_node_count=100,
                   randup_state_count=1,
                   obstacles_list = None,
                   random_state=None,
                   padding=None,
                   print_interval=1000):
    """

    :param root_state:
    :param goal_state:
    :param quadrotor:
    :param max_node_count:
    :param randup_state_count:
    :param obstacles_list: (center, radius)
    :return:
    """
    assert padding is not None
    if random_state is None:
        random_state = np.random.RandomState()
    # Setup Simulator
    forward_dynamics_sim = quadrotor.simulate_forward
    max_action = np.array([4., 4.])
    # TODO: tune values
    sample_state_highs = np.array([15, 7, 20., 20.])
    sample_state_lows = np.array([-3, -7, -20., -20.])
    goal_bias = 0.05

    def state_sampler():
        if random_state.random(1) < goal_bias:
            return goal_state
        return random_state.uniform(low=sample_state_lows, high=sample_state_highs)

    def action_sampler(x): return (random_state.rand(2)-0.5)*max_action
    # Setup RRT

    def reached_goal(states, mode):
        array_diff = states - goal_state
        dist = np.linalg.norm((goal_distance_scaling_array*array_diff)[:,:2], axis=1)
        return np.all(dist < goal_threshold-padding), np.max(dist)

    # Local controller is just open loop
    def local_controller_creator(sample_node, sample_action,
                                 sample_next_mode, forward_dynamics_sim):
        """

        :param sample_node:
        :param sample_action: νₜ
        :param sample_next_mode:
        :param forward_dynamics_sim: Quadrotor.simulate_forward
        :return:
        """
        # First simulate the nominal dynamics forward open loop
        def local_controller(x_t):
            # The control law is uₜ = νₜ+K(xₜ−μₜ)
            u_t = sample_action + quadrotor.K@(x_t-sample_node.nominal_state)
            return u_t
        return local_controller

    # Obstacles are circles in the state space
    def is_safe(states):
        """
        For checking collisions
        :param states: nx4 states for n RandUP particles
        :return:
        """
        if obstacles_list is not None:
            for center, radius in obstacles_list:
                if np.any(np.linalg.norm(center-states[:, :2], axis=1)<radius+padding):
                    return False
        return True

    rrt = randup_rrt.RandUpRRT(root_state, contact_modes.ContinuousSystemModes.MODE,
                               forward_dynamics_sim,
                               randup_state_count,
                               state_sampler, action_sampler,
                               lambda x: contact_modes.ContinuousSystemModes.MODE,
                               distance_scaling_array=distance_scaling_array,
                               interpolation_collision_check=3,
                               random_state=random_state)
    # Plan with RandUP-RRT
    rrt.plan(reached_goal, local_controller_creator, is_safe, max_node_count,
             print_interval=print_interval)
    # TODO: plan with Lipschitz + RRT
    return rrt

def is_safe_verification(state):
    """
    For checking collisions
    :param states: nx4 states for n RandUP particles
    :return:
    """
    if obstacles_list is not None:
        for center, radius in obstacles_list:
            if np.linalg.norm(center - state[:2]) < radius:
                # print(f"State {state} collided with obstacle {center, radius}"
                #       f"distance is {np.linalg.norm(center - state[:2])}")
                return False, np.linalg.norm(center - state[:2])
    return True, 0.

def reached_goal_verification(state):
    array_diff = goal_distance_scaling_array*(state - goal_state)
    # Exclude velocity in goal check
    dist = np.linalg.norm((array_diff)[:2])
    return dist < goal_threshold, dist

def plot_quad_rollout(fig, ax, root_state, num_particles,
                      quadrotor_system_viz):
    current_state = root_state
    for p_idx in range(num_particles):
        all_states = [root_state]
        current_state = root_state
        for step, lc in enumerate(local_controllers):
            current_state = quadrotor_system_viz.simulate_forward(current_state,
                                                              lc,
                                                              particle_index=p_idx,
                                                              current_mode=contact_modes.ContinuousSystemModes.MODE)[0]
            all_states.append(current_state)
            # Check for collision
            safe, penetration = is_safe_verification(current_state)
            if not safe:
                print(f"Plan has collision on rollout {p_idx}. Penetration depth: {penetration}")
                all_states = np.array(all_states)
                ax.plot(all_states[:,0], all_states[:,1],  color='r',ls='-',alpha=0.02)
                break
        # Check for goal
        reached, goal_dist = reached_goal_verification(current_state)
        if not reached:
            print(f"Plan failed to reach goal on rollout {p_idx}. Goal distance: {goal_dist}")
            print(current_state-goal_state)
            all_states = np.array(all_states)
            ax.plot(all_states[:, 0], all_states[:, 1], color='r',ls='-',alpha=0.02)
        else:
            all_states = np.array(all_states)
            ax.plot(all_states[:, 0], all_states[:, 1],  color='g',ls='-',alpha=0.02)
    return True, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Planar quadrotor demo')
    parser.add_argument('--num_particles', type=int, default=100,
                        required=False,
                        help='Number of RandUP particles. Use 1 for RRT.')
    args = parser.parse_args()
    randup_state_count = args.num_particles
    assert 1<=randup_state_count
    print(f'Using {randup_state_count} particles.')

    seed = 1
    random_state = np.random.RandomState(seed)
    max_node_count = 50000
    root_state = np.zeros(4)
    goal_state = np.asarray([10., 0., 0., 0.])
    # Uncertainties
    # No uncertainties
    # drag_alpha_x_bounds = (0.5, 0.5)
    # drag_alpha_y_bounds = (0.5, 0.5)
    # drag_alpha_x_nom = 0.5
    # drag_alpha_y_nom = 0.5

    # With uncertainties
    drag_alpha_x_bounds = (0.35, 0.65)
    drag_alpha_y_bounds = (0.35, 0.65)
    drag_alpha_x_nom = 0.5
    drag_alpha_y_nom = 0.5

    quadrotor = Quadrotor(random_state=random_state,
                          drag_alpha_x_bounds=drag_alpha_x_bounds,
                          drag_alpha_y_bounds=drag_alpha_y_bounds,
                          drag_alpha_x_nom=drag_alpha_x_nom,
                          drag_alpha_y_nom=drag_alpha_y_nom,
                          disturbance_boundary_sample_rate=0.4
                          )
    padding = 0.3

    obstacles_list = two_obstacles

    print(f"randup_state_count: {randup_state_count}")
    print(f"obstacles_list: {obstacles_list}")
    rrt = plan_quadrotor(root_state, goal_state, quadrotor, max_node_count,
                         randup_state_count=randup_state_count,
                         obstacles_list=obstacles_list,
                         random_state=random_state,
                         padding=padding)
    actions, local_controllers, expected_modes = rrt.backtrack_actions(
        rrt.best_node)
    fig, ax = plt_rrt.visualize_quadrotor_rrt(
        rrt, goal_state, show_all_nodes=False, obstacle_list=obstacles_list,
        padding=padding,
        goal_threshold=goal_threshold)

    # quad_viz = Quadrotor(random_state=random_state,
    #                       drag_alpha_x_bounds=drag_alpha_x_bounds,
    #                       drag_alpha_y_bounds=drag_alpha_y_bounds,
    #                       drag_alpha_x_nom=drag_alpha_x_nom,
    #                       drag_alpha_y_nom=drag_alpha_y_nom,
    #                      disturbance_boundary_sample_rate=0.
    #                       )
    # plot_quad_rollout(fig, ax, root_state, 100, quad_viz)

    # Hard-coded texts
    plt.text(9.65, -1.6, "$\mathcal{X}_G$")
    plt.text(-0.9, 0, "$x_0$")
    if obstacles_list==two_obstacles:
        plt.text(2.75, -1.25, "$\mathcal{C}$")
        plt.text(6.25, 1, "$\mathcal{C}$")
    filename = f'{time.strftime("%Y%m%d-%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"saved figure to {filename}")
