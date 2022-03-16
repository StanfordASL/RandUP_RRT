import numpy as np
import enum
import warnings


class RandUpRRTNode:
    def __init__(self, states_cluster, contact_mode, parent,
                 action_from_parent, controller_from_parent,
                 children=None):
        # states cluster should be a 2D numpy array
        assert len(states_cluster.shape) == 2
        assert isinstance(states_cluster, np.ndarray)
        self.states_cluster = states_cluster
        # A single state to represent the cluster.
        self.nominal_state = np.average(self.states_cluster, axis=0)
        assert isinstance(contact_mode, enum.Enum)
        self.hash_value = None
        self.contact_mode = contact_mode
        self.parent = None
        if parent is not None:
            self.parent = parent
            self.parent.add_child(self)
        self.action_from_parent = action_from_parent
        # The controller should be a callable
        if controller_from_parent is not None:
            assert callable(controller_from_parent)
        self.controller_from_parent = controller_from_parent
        self.children = set()
        if children is not None:
            self.children.update(children)

    def __hash__(self):
        # TODO(wualbert) better implementation
        if self.hash_value is None:
            self.hash_value = hash(str(self.states_cluster) +
                                   str(self.contact_mode))
        return self.hash_value

    def __eq__(self, other):
        return hash(self) == hash(other)

    def add_child(self, child_node):
        assert isinstance(child_node, RandUpRRTNode)
        self.children.add(child_node)
        child_node.parent = self

    def get_nominal_state(self):
        return self.nominal_state
