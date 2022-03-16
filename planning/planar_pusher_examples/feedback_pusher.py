import planar_pusher.visualization.plot_planar_pusher_rrt as plt_rrt
import planning.contact_modes as contact_modes
import planning.randup_rrt as randup_rrt
import planar_pusher.param as param
import planar_pusher.planar_pusher_example_worlds as example_worlds
import planar_pusher.planar_pusher_controllers as controllers
import utils.umath as umath

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from pybullet_utils import bullet_client
import pybullet as p
import warnings

distance_scaling_array = np.array([1., 1., 0.2, 0.2, 0.2, 0.04])
# Normalize
distance_scaling_array /= np.linalg.norm(distance_scaling_array)
goal_distance_scaling_array = np.array([1., 1., 0.05, 0.5, 0.5, 0.05])
goal_distance_scaling_norm = np.linalg.norm(goal_distance_scaling_array)
goal_distance_scaling_array /= np.linalg.norm(goal_distance_scaling_array)
goal_threshold = 0.5


def plan_planar_pusher(root_state, goal_state,
                       planar_pusher,
                       max_node_count=100,
                       randup_state_count=1,
                       noise_magnitude=1.,
                       random_state=None):
    # Setup Simulator
    gains = np.asarray([300., -20.,
                        100.,  150.,
                        -10.,  50.]
                       )
    f_max = 8.
    displacement_max = 3.
    displacement_min = 0.5
    goal_bias = 0.25
    # Setup RRT
    half_aperature = \
        planar_pusher.objects[planar_pusher.box_object_index].\
        half_friction_cone_aperture
    azimuth_max = np.pi
    yaw_max = half_aperature  # np.pi/2.
    print('azimuth yaw', azimuth_max, yaw_max)
    projection_controller = controllers.ProjectionController(
        half_aperature,
        gains, f_max,
        abs(param.GRAVITY *
            planar_pusher.objects[planar_pusher.box_object_index].mu)
    )

    def local_controller_creator(sample_node, sample_action,
                                 sample_next_mode, forward_dynamics_sim):
        def local_controller(current_state):
            return projection_controller.compute_action(current_state,
                                                        sample_action)
        return local_controller

    sampled_actions = []
    sampled_states = []

    def action_sampler(sampled_node):
        # go to a random state with zero velocity
        if random_state.rand() < goal_bias:
            goal_direction = goal_state - sampled_node.nominal_state
            goal_direction[param.BOX_STATE_THETA] = \
                umath.angle_diff_wrapped(
                    goal_state[param.BOX_STATE_THETA],
                    sampled_node.nominal_state[param.BOX_STATE_THETA])
            norm = np.linalg.norm(goal_direction)
            normalized_goal_direction = goal_direction/norm

            target_state = sampled_node.nominal_state + \
                normalized_goal_direction*min(norm,
                                              displacement_max)
            target_state[param.BOX_STATE_THETA] %= 2*np.pi
            return target_state
        sample_action = np.zeros(6)
        sample_action[param.BOX_STATE_THETA] = (random_state.normal(0., yaw_max) +
                                                sampled_node.nominal_state[param.BOX_STATE_THETA]) % (2*np.pi)

        random_displacement = random_state.uniform(displacement_min,
                                                displacement_max)
        random_azimuth = random_state.normal(0., azimuth_max)
        R_W_target = umath.get_R_AB(
            random_azimuth+sampled_node.nominal_state[param.BOX_STATE_THETA])
        sample_action[:param.BOX_STATE_THETA] = sampled_node.nominal_state[:2] + \
            R_W_target[:, 0]*random_displacement
        sampled_actions.append(sample_action)
        return sample_action

    def forward_dynamics_sim(starting_state, local_controller, *args, **kwargs):
        return planar_pusher.simulate_forward_and_check_collision(
            starting_state, local_controller, noise_magnitude=noise_magnitude
        )
    # In this parameterization, the action is the coefficient of the
    # friction cone basis ranging from 0~max
    sample_state_highs = np.array([5., 5., np.pi/2., 1., 1., 1.])
    sample_state_lows = -np.array([15., 10., np.pi/2, 1., 1., 1.])
    def state_sampler():
        if random_state.rand() < goal_bias:
            sample = goal_state
        else:
            sample = random_state.uniform(
                low=sample_state_lows, high=sample_state_highs)
        sampled_states.append(sample)
        return sample

    def reached_goal(state, mode):
        array_diff = state - goal_state
        array_diff[:, param.BOX_STATE_THETA] = umath.angle_diff_wrapped(state[:, param.BOX_STATE_THETA],
                                                                        goal_state[param.BOX_STATE_THETA])
        dist = np.linalg.norm(goal_distance_scaling_array*array_diff, axis=1)
        return np.all(dist < goal_threshold), np.max(dist)

    def is_safe(states):
        return True

    rrt = randup_rrt.RandUpRRT(root_state, contact_modes.ContinuousSystemModes.MODE,
                               forward_dynamics_sim,
                               randup_state_count,
                               state_sampler, action_sampler,
                               lambda x: contact_modes.ContinuousSystemModes.MODE,
                               distance_scaling_array=distance_scaling_array,
                               wrap_indices=np.array([param.BOX_STATE_THETA]),
                               random_state=random_state)
    rrt.plan(reached_goal, local_controller_creator, is_safe, max_node_count)
    # sampled_actions = np.asarray(sampled_actions)
    # sample_states = np.asarray(sampled_states)
    # plt.scatter(sampled_actions[:, 0], sampled_actions[:, 1], color='r', s=0.5)
    # plt.scatter(sample_states[:, 0], sample_states[:, 1], color='b', s=0.5)
    # plt.show()
    return rrt


def playback_plan_from_scratch(root_state, local_controllers, world_type,
                               noise_magnitude,
                               goal_pose=None,
                               playback_p=None,
                               stop_at_collision=True,
                               sleep=False):
    if playback_p is None:
        playback_p = bullet_client.BulletClient(connection_mode=p.GUI)
    planar_pusher = example_worlds.create_planar_pusher_world(
        world_type,
        bullet_p=playback_p,
        goal_pose=goal_pose)
    return playback_plan_given_planar_pusher(root_state, local_controllers,
                                             planar_pusher, noise_magnitude,
                                             stop_at_collision,
                                             sleep=sleep)


def playback_plan_given_planar_pusher(root_state, local_controllers,
                                      planar_pusher,
                                      noise_magnitude,
                                      stop_at_collision=True,
                                      sleep=False):
    # create the goal object
    planar_pusher.set_state(root_state)
    starting_state = root_state
    return_traj=[root_state]
    for local_controller in local_controllers:
        starting_state, _, has_collision = \
            planar_pusher.simulate_forward_and_check_collision(
                starting_state, local_controller, noise_magnitude=noise_magnitude,
            sleep=sleep)
        return_traj.append(starting_state)
        if has_collision:
            print(
                f"{param.BColors.WARNING}"
                f"Collision detected!{param.BColors.ENDC}")
            if stop_at_collision:
                return False, return_traj
    return True, return_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feedback pusher demo')
    parser.add_argument('--num_particles', type=int, default=20,
                        required=False,
                        help='Number of RandUP particles. Use 1 for RRT.')
    args = parser.parse_args()
    randup_state_count = args.num_particles
    assert 1<=randup_state_count
    print(f'Using {randup_state_count} particles.')
    seed = 3
    random_state = np.random.RandomState(seed)
    max_node_count = 5000
    root_state = np.zeros(6)
    root_state[0] = 4.
    goal_state = np.asarray([-15., 0., 0., 0., 0., 0.])
    noise_magnitude = 1.
    playback_verification_count = 10
    gui_playback_count = 100
    world_type = example_worlds.NarrowPassageWorld()
    print(f'Parameters: \nmax node {max_node_count}'
          f'\nroot state {root_state} \ngoal state {goal_state}'
          f'\nrandup state count {randup_state_count}'
          f'\nnoise magnitude {noise_magnitude}'
          f'\nworld type {world_type}\n')
    planar_pusher = example_worlds.create_planar_pusher_world(
        world_type, goal_pose=goal_state[:3], np_random=random_state)
    rrt = plan_planar_pusher(root_state, goal_state, planar_pusher,
                             max_node_count,
                             randup_state_count=randup_state_count,
                             noise_magnitude=noise_magnitude,
                             random_state=random_state)
    node_count = len(rrt.state_to_node_map)
    actions, local_controllers, expected_modes = rrt.backtrack_actions(
        rrt.best_node)

    # plt.show()

    # playback plan
    playback_p = bullet_client.BulletClient(connection_mode=p.DIRECT)
    execution_count = 0
    collided_count = 0
    fig, ax = plt.subplots()
    for i in range(playback_verification_count):
        execution_count += 1
        no_collision, return_traj = playback_plan_from_scratch(root_state,
                                                               local_controllers, world_type,
                                                  noise_magnitude,
                                                  goal_pose=goal_state[:3],
                                                  playback_p=playback_p)
        collided_count += int(not(no_collision))
        # add the returned trajectory to the plot
        if no_collision:
            color='teal'
        else:
            color='magenta'
        plt_rrt.visualize_planar_pusher_trajectory(return_traj, fig, ax, color,
                                                   alpha=0.1)
    # Plotting
    fig, ax = plt_rrt.visualize_planar_pusher_continuous_rrt(
        rrt, goal_state, show_all_nodes=True, world_type=world_type,
        goal_threshold=goal_distance_scaling_norm * goal_threshold,
        fig=fig, ax=ax)
    print(f'{param.BColors.OKBLUE}'
          f'Collided: {collided_count}/{execution_count} = '
          f'{collided_count/execution_count*100.}%'
          f'{param.BColors.ENDC}')

    filename = f'{time.strftime("%Y%m%d-%H%M%S")}_'\
        f'noise_{noise_magnitude}_randup_{randup_state_count}_nodes_{node_count}.png'
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f'Saved to {filename}')
    
    for g in range(gui_playback_count):
        playback_p = bullet_client.BulletClient(connection_mode=p.GUI)
        execution_count = 0
        collided_count = 0
        for i in range(100):
            execution_count += 1
            no_collision, return_traj = playback_plan_from_scratch(root_state, local_controllers, world_type,
                                                      noise_magnitude,
                                                      goal_pose=goal_state[:3],
                                                      playback_p=playback_p,
                                                      stop_at_collision=False,
                                                      sleep=False)
            collided_count += int(not(no_collision))
            print(f'{param.BColors.OKCYAN}'
                  f'Collided: {collided_count}/{execution_count} = '
                  f'{collided_count / execution_count * 100.}%'
                  f'{param.BColors.ENDC}')
