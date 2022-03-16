import planning.randup_rrt_node as randup_rrt_node
import planning.contact_modes as contact_modes
import planning.state_tree as state_tree
from collections import deque
import numpy as np
from timeit import default_timer


class RandUpRRT:
    def __init__(self, root_state, root_mode, forward_dynamics_sim,
                 randup_state_count,
                 state_sampler, action_sampler, mode_sampler,
                 modes_enum=contact_modes.ContinuousSystemModes,
                 random_seed=None,
                 wrap_indices=np.zeros(0),
                 distance_scaling_array=None,
                 random_state=None,
                 interpolation_collision_check=None):
        self.wrap_indices = wrap_indices
        self.root_state = np.ndarray.flatten(root_state)
        self.root_mode = root_mode
        self.state_dim = self.root_state.shape[0]
        self.node_count = 0
        # For sampling
        self.all_nodes_list = []
        self.modes_enum = modes_enum
        # Construct the root node
        self.root_node = self.construct_and_add_rrt_node(
            np.atleast_2d(self.root_state),
            self.root_mode,
            None, None, None)
        self.forward_dynamics_sim = forward_dynamics_sim
        self.randup_state_count = randup_state_count

        self.distance_scaling_array = distance_scaling_array
        if self.distance_scaling_array is None:
            self.distance_scaling_array = np.ones(
                self.root_state.shape).flatten()
        self.state_tree = state_tree.StateTree(self.distance_scaling_array,
                                               self.wrap_indices)
        self.state_sampler = state_sampler
        self.action_sampler = action_sampler
        self.mode_sampler = mode_sampler
        self.best_node = None
        # TODO(wualbert): sampling with proper voronoi bias
        if random_state is None:
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = random_state
        self.goal_node = None
        self.state_tree.insert(self.root_node.nominal_state)
        self.state_to_node_map = dict()
        self.state_to_node_map[hash(
            str(self.root_node.nominal_state))] = [self.root_node]
        self.interpolation_collision_check = interpolation_collision_check

    def plan(self, reached_goal, local_controller_creator, is_safe,
             max_node_count, print_interval=50):
        assert not np.isinf(max_node_count)
        start_time = default_timer()
        best_goal_dist = np.inf
        # Check if the root is in the goal (trivial condition)
        discarded_unsafe_node_count = 0
        success, goal_dist = reached_goal(self.root_node.states_cluster,
                                          self.root_node.contact_mode)
        if success:
            self.goal_node = self.root_node
        if best_goal_dist > goal_dist:
            best_goal_dist = goal_dist
            self.best_node = self.root_node
        while self.goal_node is None and max_node_count >= self.node_count:
            # First sample node
            # TODO(wualbert): better data structure
            sample_state = self.state_sampler()
            sample_node_state = self.state_tree.k_nearest_states(sample_state)
            # TODO(wualbert): better biasing of modes
            sample_node = self.random_state.choice(self.state_to_node_map[hash(
                str(sample_node_state.flatten()))])
            # Sample contact mode
            sample_next_mode = self.mode_sampler(sample_node)
            # Then sample action
            sample_action = self.action_sampler(sample_node)
            # Create the local controller
            sample_controller = local_controller_creator(
                sample_node,
                sample_action,
                sample_next_mode,
                self.forward_dynamics_sim)
            # step forward with the
            resulting_states, resulting_modes, has_collision = self.step_forward(
                sample_node,
                sample_action,
                sample_controller,
                sample_next_mode)
            if has_collision:
                continue
            if len(self.wrap_indices) > 0:
                resulting_states[:, self.wrap_indices] %= 2*np.pi
            # Try to extend the tree using the action
            if not is_safe(resulting_states):
                # If the action is unsafe, discard and redo
                discarded_unsafe_node_count += 1
                if discarded_unsafe_node_count % print_interval == 0:
                    print(f'discarded {discarded_unsafe_node_count} nodes')
                continue
            if self.interpolation_collision_check:
                # Check using interpolation
                interval = (resulting_states-sample_node.states_cluster)/self.interpolation_collision_check
                failed_check = False
                for i in range(1, self.interpolation_collision_check):
                    check_state = sample_node.states_cluster + interval*i
                    if not is_safe(check_state):
                        # If the action is unsafe, discard and redo
                        discarded_unsafe_node_count += 1
                        if discarded_unsafe_node_count % print_interval == 0:
                            print(f'discarded {discarded_unsafe_node_count} nodes')
                        failed_check=True
                        break
                if failed_check:
                    continue
            # The mode transition must be deterministic
            if not np.all(resulting_modes == resulting_modes[0]):
                # TODO(wualbert): bookkeeping
                continue
            # If the action is safe, create a node
            new_node = self.construct_and_add_rrt_node(resulting_states,
                                                       resulting_modes[0],
                                                       sample_node,
                                                       sample_action,
                                                       sample_controller)
            # Add new node to the state tree
            self.state_tree.insert(new_node.nominal_state)
            state_hash = hash(str(new_node.nominal_state))
            if state_hash not in self.state_to_node_map:
                self.state_to_node_map[hash(
                    str(new_node.nominal_state))] = [new_node]
            else:
                self.state_to_node_map[hash(
                    str(new_node.nominal_state))].append(new_node)
            if self.node_count % print_interval == 0:
                print(f'generated {self.node_count} nodes,'
                      f'best_goal_dist = {best_goal_dist}')
            # If the goal contains the resulting state cluster,
            # the algorithm terminates
            success, goal_dist = reached_goal(new_node.states_cluster,
                                              new_node.contact_mode)
            if success:
                self.goal_node = new_node
            if best_goal_dist > goal_dist:
                best_goal_dist = goal_dist
                self.best_node = new_node
        print(f'\nnode_count = {self.node_count}\n'
              f'discarded_unsafe_node_count = {discarded_unsafe_node_count}\n'
              f'runtime(sec) = {default_timer()-start_time}\n'
              f'best_goal_dist = {best_goal_dist}')
        if self.goal_node is not None:
            self.goal_node = self.goal_node
            return self.backtrack_actions(self.goal_node)
        else:
            return None

    def step_forward(self, node, control_action, local_controller,
                     desired_mode):
        # Using the forward dynamics simulator
        # Compute the resulting states
        # TODO(wualbert): better sampling strategy
        random_indices = self.random_state.randint(
            node.states_cluster.shape[0],
            size=self.randup_state_count)
        resulting_states = np.zeros([self.randup_state_count, self.state_dim])
        resulting_modes = np.empty([self.randup_state_count], dtype=object)
        # TODO(wualbert): vectorize the computation
        cluster_has_collision = False
        for i, starting_state in \
                enumerate(node.states_cluster[random_indices]):
            resulting_state, resulting_mode, has_collision = \
                self.forward_dynamics_sim(
                    starting_state, local_controller,
                    current_mode=node.contact_mode,
                    particle_index=random_indices[i])
            resulting_states[i, :] = resulting_state
            resulting_modes[i] = resulting_mode
            if has_collision:
                cluster_has_collision = True
                resulting_states = None
                resulting_modes = None
                break
        return resulting_states, resulting_modes, cluster_has_collision

    def construct_and_add_rrt_node(self, states_cluster, contact_mode,
                                   parent_node,
                                   action_fom_parent, controller_from_parent):
        new_node = randup_rrt_node.RandUpRRTNode(states_cluster, contact_mode,
                                                 parent_node,
                                                 action_fom_parent,
                                                 controller_from_parent)
        self.all_nodes_list.append(new_node)
        self.node_count += 1
        return new_node

    def backtrack_actions(self, successful_node):
        actions = deque()
        local_controllers = deque()
        expected_modes = deque()
        node = successful_node
        while node is not None:
            actions.append(node.action_from_parent)
            local_controllers.append(node.controller_from_parent)
            expected_modes.append(node.contact_mode)
            node = node.parent
        actions.reverse()
        local_controllers.reverse()
        expected_modes.reverse()
        a0 = actions.popleft()
        c0 = local_controllers.popleft()
        assert a0 is None
        assert c0 is None
        assert len(actions) == len(local_controllers)
        assert len(actions) == len(expected_modes) - 1
        return actions, local_controllers, expected_modes
