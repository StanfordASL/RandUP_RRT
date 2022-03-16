import numpy as np
from rtree import index


class StateTree:
    """
    Wrapper for rtree for quick nearest-state querying. Supports
    angle wrap-around and scaling different distance indices.
    """

    def __init__(self, distance_scaling_array,
                 wrap_indices):
        """
        @param distance_scaling_array: A numpy array that will be used
        for computing the distance between two generalized positions
        dist(q1, q2) = ∑ᵢ distance_scaling_array(i) * (q1(i) - q2(i))².
        @param wrap_indices: Indices that will consider angle
        wrap-arounds. Set to None if nothing is wrapped.
        """
        if len(wrap_indices) > 0:
            assert(isinstance(wrap_indices, np.ndarray))
            assert(np.max(wrap_indices) < distance_scaling_array.shape[0])
        else:
            wrap_indices = np.zeros(0)
        self.state_id_to_state = {}
        self.state_tree_p = index.Property()
        self.state_idx = None
        self.distance_scaling_array = distance_scaling_array
        self.wrapped_id_to_original_id = {}
        self.wrap_indices = wrap_indices
        if len(wrap_indices) > 0:
            wrap_multipliers = np.tile(np.asarray(
                [-1., 0., 1.]), self.wrap_indices.shape[0]).\
                reshape([self.wrap_indices.shape[0], -1])
            self.possible_wraps = np.array(np.meshgrid(
                *wrap_multipliers)).T.reshape(-1, self.wrap_indices.shape[0])\
                * 2*np.pi
        else:
            self.possible_wraps = None

        self.wrapped_id_to_wrapped_state = {}

        # initialize rtree
        self.state_tree_p.dimension = distance_scaling_array.shape[0]
        # internally, rtree points are stored as bounding boxes
        # (min, max). To store a point p, we set the bounding box to
        # (p, p)
        self.state_idx = index.Index(properties=self.state_tree_p)
        self.dim = distance_scaling_array.shape[0]

    def insert(self, state, state_id=None):
        """
        Insert a state into the data structure. Internally, all possible
        wrap-around states are inserted and memoized.
        @param state: manipuland state to insert. Same shape as
        distance_scaling_array
        @param state_id: ID of the state to insert.
        @return: None
        """
        if state_id is None:
            state_id = hash(str(state))
        assert(state.shape == self.distance_scaling_array.shape)
        # require the state to be insert to be "unwrapped"
        self.state_id_to_state[state_id] = state
        if len(self.wrap_indices) > 0:
            assert(np.all(state[self.wrap_indices] ==
                          state[self.wrap_indices] % (2.*np.pi)))
            for wrap in self.possible_wraps:
                angle_diff = np.zeros(self.distance_scaling_array.shape[0])
                np.put(angle_diff, self.wrap_indices, wrap)
                scaled_state = np.multiply(np.sqrt(self.distance_scaling_array),
                                           state + angle_diff)
                wrapped_id = hash(str(scaled_state))
                # internally, rtree points are stored as bounding boxes
                # (min, max). To store a point p, we set the bounding box to
                # (p, p)
                self.state_idx.insert(wrapped_id, np.tile(scaled_state, 2))
                self.wrapped_id_to_original_id[wrapped_id] = state_id
                self.wrapped_id_to_wrapped_state[wrapped_id] = state + angle_diff
        else:
            self.state_idx.insert(state_id, np.tile(state, 2))

    def k_nearest_states(self, query_state, k=1):
        """
        Find the k nearest states to query_state
        @param query_state: query state
        @param k: maximum number of nearest states to return
        @return: 2D array of (state shape, k)
        """
        # k*self.possible_wraps.shape[0] is for preventing finding
        # wrapped states
        # for querying, we are also using a "box". Hence we query
        # with (x, x)
        if len(self.wrap_indices) == 0:
            np.testing.assert_equal(query_state.shape, (self.dim,))
            nearest_states_ids = self.state_idx.nearest(
                np.tile(np.multiply(np.sqrt(self.distance_scaling_array),
                                    query_state), 2))
            nearest_states = []
            for nearest_state_id in nearest_states_ids:
                nearest_states.append(self.state_id_to_state[nearest_state_id])
            return np.asarray(nearest_states).T[:, :k]

        query_state[self.wrap_indices] %= 2*np.pi
        np.testing.assert_equal(query_state.shape, (self.dim,))
        nearest_wrapped_state_ids = self.state_idx.nearest(
            np.tile(np.multiply(np.sqrt(self.distance_scaling_array),
                                query_state), 2),
            k*self.possible_wraps.shape[0])
        nearest_states = []
        used_state_ids = set()
        for nearest_wrapped_state_id in nearest_wrapped_state_ids:
            # convert back to original state id
            nearest_state_id = \
                self.wrapped_id_to_original_id[nearest_wrapped_state_id]
            # convert to raw state
            if nearest_state_id not in used_state_ids:
                nearest_states.append(self.state_id_to_state[nearest_state_id])
                used_state_ids.add(nearest_state_id)
        return np.asarray(nearest_states).T[:, :k]

    def states_in_r_box(self, query_state, r):
        '''
        Find the states within the bounding hyperbox with side length 2r
        to the query states.
        r >= sqrt(distance_scaling_array(i)) * |q1(i) - q2(i)|
        This should return states that are within sqrt(N)*r for N-D states
        @param query_state: query state
        @param r: maximum bounding box side length.
        @return: 2D numpy array of shape (*, state shape), which are the
        states in the hyperbox.
        '''
        assert (r >= 0.)
        scaled_state = np.multiply(np.sqrt(self.distance_scaling_array),
                                   query_state)
        wrapped_state_ids_in_box = self.state_idx.intersection(
            np.hstack([scaled_state-r, scaled_state+r]))
        states_in_box = []
        used_state_ids = set()
        for nearest_wrapped_state_id in wrapped_state_ids_in_box:
            # convert back to original state id
            nearest_state_id = \
                self.wrapped_id_to_original_id[nearest_wrapped_state_id]
            # convert to raw state
            if nearest_state_id not in used_state_ids:
                states_in_box.append(self.state_id_to_state[nearest_state_id])
                used_state_ids.add(nearest_state_id)
        return np.asarray(states_in_box).T
