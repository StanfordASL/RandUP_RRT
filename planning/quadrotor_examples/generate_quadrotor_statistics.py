import plan_quadrotor
from quadrotor_planar.quadrotor import Quadrotor
import planning.contact_modes as contact_modes

import time
import numpy as np

def quadrotor_monte_carlo_verification(quadrotor_system,
                                       root_state,
                                       actions, local_controllers,
                                       is_safe,
                                       reached_goal,
                                       num_particles=1000):
    current_state = root_state
    for p_idx in range(num_particles):
        current_state = root_state
        for step, lc in enumerate(local_controllers):
            current_state = quadrotor_system.simulate_forward(current_state,
                                                              lc,
                                                              particle_index=p_idx,
                                                              current_mode=contact_modes.ContinuousSystemModes.MODE)[0]
            # Check for collision
            safe, penetration = is_safe(current_state)
            if not safe:
                print(f"Plan has collision on rollout {p_idx}. Penetration depth: {penetration}")
                return False, penetration
        # Check for goal
        reached, goal_dist = reached_goal(current_state)
        if not reached:
            print(f"Plan failed to reach goal on rollout {p_idx}. Goal distance: {goal_dist}")
            print(current_state-goal_state)
            return False, goal_dist
    return True, None


if __name__ == "__main__":
    # No randup states
    randup_state_count = 1
    padding = 0.5

    # Normal randup state
    # randup_state_count = 100
    # padding = 0.3

    obstacles_list = plan_quadrotor.two_obstacles

    repeats = 50
    seed = 1
    random_state = np.random.RandomState(seed)
    max_node_count = 12000
    verification_particles = 10000
    root_state = np.zeros(4)
    goal_state = np.asarray([10., 0., 0., 0.])

    # Uncertainties
    drag_alpha_x_bounds = (0.35, 0.65)
    drag_alpha_y_bounds = (0.35, 0.65)
    drag_alpha_x_nom = 0.5
    drag_alpha_y_nom = 0.5


    distance_scaling_array = np.array([1., 1., 0.1, 0.1])
    goal_bias = 0.05
    # Normalize
    distance_scaling_array /= np.linalg.norm(distance_scaling_array)
    print("Randup state count : ", randup_state_count)
    print("Padding : ", padding)
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
        array_diff = plan_quadrotor.goal_distance_scaling_array*(state - goal_state)
        # Exclude velocity in goal check
        dist = np.linalg.norm((array_diff)[:2])
        return dist < plan_quadrotor.goal_threshold, dist

    def get_quadrotor_system(random_state=None,
                             disturbance_boundary_sample_rate=0.4):
        return Quadrotor(random_state=random_state,
                          drag_alpha_x_bounds=drag_alpha_x_bounds,
                          drag_alpha_y_bounds=drag_alpha_y_bounds,
                          drag_alpha_x_nom=drag_alpha_x_nom,
                          drag_alpha_y_nom=drag_alpha_y_nom,
                         disturbance_boundary_sample_rate=disturbance_boundary_sample_rate
                          )

    # timing information
    runtimes = np.zeros(repeats)
    node_counts = np.zeros(repeats)
    successes = 0
    valids = 0
    violation_distances = []
    for i in range(repeats):
        t0 = time.time()
        quadrotor = get_quadrotor_system(random_state, disturbance_boundary_sample_rate=0.4)
        rrt = plan_quadrotor.plan_quadrotor(root_state, goal_state, quadrotor, max_node_count,
                                            randup_state_count=randup_state_count,
                                            random_state=random_state,
                                            obstacles_list=obstacles_list,
                                            padding=padding)
        actions, local_controllers, expected_modes = rrt.backtrack_actions(
            rrt.best_node)
        t1 = time.time()
        runtimes[i] = t1-t0
        node_counts[i] = rrt.node_count
        successes += (rrt.goal_node is not None)

        # Verification alpha should match the ground truth (uniform distribution)
        verification_quadrotor = get_quadrotor_system(random_state,
                                                      disturbance_boundary_sample_rate=0.)
        print("Verifying...")
        is_valid, violation_distance = quadrotor_monte_carlo_verification(verification_quadrotor,
                                                      rrt.root_state,
                                                      actions,
                                                      local_controllers,
                                                      is_safe_verification,
                                                      reached_goal_verification,
                                                      verification_particles)
        if not is_valid and not np.isnan(violation_distance):
            violation_distances.append(violation_distance)
        valids += int(is_valid)
        print(f"=============================================================")
        print(f"Completed run {i} in {runtimes[i]} sec")
        print(f"Generated {node_counts[i]} nodes, "
              f"found path={rrt.goal_node is not None}")
        print(f"Plan valid: {is_valid}")
        print(f"=============================================================")

    print(f"Runtime stats: {np.average(runtimes)} ± {np.std(runtimes)}")
    print(f"runtime max: {np.max(runtimes)}, min: {np.min(runtimes)}")
    print(f"Generated node count: {np.average(node_counts)} ± {np.std(node_counts)}")
    print(f"node count max: {np.max(node_counts)}, min: {np.min(node_counts)}")
    print(f"Planning success rate: {successes/repeats}")
    print(f"Obtained plan valid rate: {valids/successes}")
    violation_distances = np.array(violation_distances)
    print(f"Invalid plan violation distance: {np.average(violation_distances)} ± {np.std(violation_distances)}")
