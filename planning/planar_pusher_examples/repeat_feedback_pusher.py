from planning.planar_pusher_examples.feedback_pusher import *

if __name__ == "__main__":
    seed = 0
    max_node_count = 2000
    root_state = np.zeros(6)
    root_state[0] = 4.
    goal_state = np.asarray([-12., 0., 0., 0., 0., 0.])
    randup_state_count = 1
    noise_magnitude = 4.
    playback_verification_count = 1
    gui_playback_count = 0
    repeats = 10
    world_type = example_worlds.NarrowPassageWorld()
    print(f'Parameters: \nmax node {max_node_count}'
          f'\nroot state {root_state} \ngoal state {goal_state}'
          f'\nrandup state count {randup_state_count}'
          f'\nnoise magnitude {noise_magnitude}'
          f'\nworld type {world_type}\n')
    planar_pusher = example_worlds.create_planar_pusher_world(
        world_type, goal_pose=goal_state[:3])
    rrt = plan_planar_pusher(root_state, goal_state, planar_pusher,
                             max_node_count,
                             randup_state_count=randup_state_count,
                             noise_magnitude=noise_magnitude,
                             random_seed=seed)
    goal_dists = []
    tree_size = []
    for i in range(repeats):
        actions, local_controllers, expected_modes = rrt.backtrack_actions(
            rrt.best_node)
    # Plotting
    plt_rrt.visualize_planar_pusher_continuous_rrt(
        rrt, goal_state, show_all_nodes=True, world_type=world_type)
    # plt.show()
    plt.savefig(f'{time.strftime("%Y%m%d-%H%M%S")}_'
                f'noise_{noise_magnitude}_randup_{randup_state_count}.png', dpi=300)

    # playback plan
    playback_p = bullet_client.BulletClient(connection_mode=p.DIRECT)
    execution_count = 0
    collided_count = 0
    for i in range(playback_verification_count):
        execution_count += 1
        no_collision = playback_plan_from_scratch(root_state, local_controllers, world_type,
                                                  noise_magnitude,
                                                  goal_pose=goal_state[:3],
                                                  playback_p=playback_p)
        collided_count += int(not(no_collision))
    print(f'{param.BColors.OKBLUE}'
          f'Collided: {collided_count}/{execution_count} = '
          f'{collided_count/execution_count*100.}%'
          f'{param.BColors.ENDC}')

    for g in range(gui_playback_count):
        playback_p = bullet_client.BulletClient(connection_mode=p.GUI)
        execution_count = 0
        collided_count = 0
        for i in range(100):
            execution_count += 1
            no_collision = playback_plan_from_scratch(root_state, local_controllers, world_type,
                                                      noise_magnitude,
                                                      goal_pose=goal_state[:3],
                                                      playback_p=playback_p,
                                                      stop_at_collision=False)
            collided_count += int(not(no_collision))
            print(f'{param.BColors.OKCYAN}'
                  f'Collided: {collided_count}/{execution_count} = '
                  f'{collided_count / execution_count * 100.}%'
                  f'{param.BColors.ENDC}')
