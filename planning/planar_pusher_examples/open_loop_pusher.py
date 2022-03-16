import planar_pusher.planar_pusher_world as planar_pusher_world
import planar_pusher.visualization.plot_planar_pusher_rrt as plt_rrt
import planning.contact_modes as contact_modes
import planning.randup_rrt as randup_rrt
import planar_pusher.param as param
import planar_pusher.planar_pusher_example_worlds as example_worlds
import utils.umath as umath

import matplotlib.pyplot as plt
import numpy as np
import time
from pybullet_utils import bullet_client
import pybullet as p
import warnings


def plan_planar_pusher(root_state, goal_state,
                       planar_pusher,
                       max_node_count=100,
                       randup_state_count=1,
                       obstacle_poses=None):
    # Setup Simulator
    forward_dynamics_sim = planar_pusher.simulate_forward_and_check_collision
    # In this parameterization, the action is the coefficient of the
    # friction cone basis ranging from 0~max
    max_action = 10.
    sample_state_highs = np.array([10., 10., np.pi, 3., 3., 3.])
    sample_state_lows = -np.array([10., 10., np.pi, 3., 3., 3.])
    distance_scaling_array = np.array([1., 1., 10., 1., 1., 10.])
    # Normalize
    distance_scaling_array /= np.linalg.norm(distance_scaling_array)
    goal_distance_scaling_array = np.array([1, 1, 2., 1., 1., 1.])
    goal_distance_scaling_array /= np.linalg.norm(goal_distance_scaling_array)

    def state_sampler():
        return np.random.uniform(low=sample_state_lows, high=sample_state_highs)

    def action_sampler(x): return np.random.rand(2)*max_action
    # Setup RRT
    goal_threshold = 0.5

    def reached_goal(state, mode):
        array_diff = state - goal_state
        array_diff[:, param.BOX_STATE_THETA] = umath.angle_diff_wrapped(state[:, param.BOX_STATE_THETA],
                                                                        goal_state[param.BOX_STATE_THETA])
        dist = np.linalg.norm(goal_distance_scaling_array*array_diff, axis=1)
        return np.all(dist < goal_threshold), np.max(dist)

    def local_controller_creator(sample_node, sample_action,
                                 sample_next_mode, forward_dynamics_sim):
        return lambda x: np.hstack([[0.], sample_action])

    def is_safe(states):
        return True

    rrt = randup_rrt.RandUpRRT(root_state, contact_modes.ContinuousSystemModes.MODE,
                               forward_dynamics_sim,
                               randup_state_count,
                               state_sampler, action_sampler,
                               lambda x: contact_modes.ContinuousSystemModes.MODE,
                               distance_scaling_array=distance_scaling_array,
                               wrap_indices=np.array([param.BOX_STATE_THETA]))
    rrt.plan(reached_goal, local_controller_creator, is_safe, max_node_count)
    return rrt


def playback_planar_pusher(root_state, local_controllers, world_type,
                           goal_pose=None,
                           playback_p=None):
    if playback_p is None:
        playback_p = bullet_client.BulletClient(connection_mode=p.GUI)
    planar_pusher = example_worlds.create_planar_pusher_world(world_type,
                                                              bullet_p=playback_p,
                                                              goal_pose=goal_pose)
    # create the goal object
    planar_pusher.set_state(root_state)
    starting_state = root_state
    for local_controller in local_controllers:
        starting_state, _, has_collision = planar_pusher.simulate_forward_and_check_collision(
            starting_state, local_controller)
        if has_collision:
            warnings.warn("Collision detected in plan playback.")


if __name__ == "__main__":
    np.random.seed(0)
    max_node_count = 10000
    root_state = np.zeros(6)
    goal_state = np.asarray([-8., 0., 0., 0., 0., 0.])
    randup_state_count = 1
    world_type = example_worlds.SimpleObstacleWorld()
    planar_pusher = example_worlds.create_planar_pusher_world(
        world_type, goal_pose=goal_state[:3])
    rrt = plan_planar_pusher(root_state, goal_state, planar_pusher, max_node_count,
                             randup_state_count=randup_state_count)
    actions, local_controllers, expected_modes = rrt.backtrack_actions(
        rrt.best_node)
    plt_rrt.visualize_planar_pusher_continuous_rrt(
        rrt, goal_state, show_all_nodes=True, world_type=world_type)
    plt.show()
    plt.savefig(f'{time.strftime("%Y%m%d-%H%M%S")}.png', dpi=300)

    # playback plan
    playback_p = bullet_client.BulletClient(connection_mode=p.GUI)
    while(1):
        playback_planar_pusher(root_state, local_controllers, world_type,
                               goal_pose=goal_state[:3],
                               playback_p=playback_p)
