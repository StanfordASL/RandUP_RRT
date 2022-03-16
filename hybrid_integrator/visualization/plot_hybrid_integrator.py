import utils.umath as umath
from collections import deque
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import collections as mc
from matplotlib.patches import Polygon, Circle
import numpy as np
import scipy.spatial
import warnings
import scipy.spatial.qhull
import time

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 24})

def plot_convex_hull(states, ax, *args, **kwargs):
    if states.shape[0] < 3:
        return
    try:
        hull = scipy.spatial.ConvexHull(states)
    except scipy.spatial.qhull.QhullError:
        warnings.warn("Qhull error")
        return
    # np.hstack([hull.vertices, [hull.vertices[0]]])
    hull_vertices = hull.vertices
    p = Polygon(states[hull_vertices], *args, **kwargs)
    ax.add_patch(p)


def visualize_hybrid_integrator_rrt(rrt, goal_state=None,
                                    show_all_nodes=False,
                                    ax=None, fig=None,
                                    world=None,
                                    goal_radius=None):
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    # visualize start
    plot_state = rrt.root_node.nominal_state
    ax.scatter([plot_state[0]], [plot_state[2]], c='r', s=7)
    if goal_state is not None:
        ax.scatter([plot_state[0]], [plot_state[2]], c='g', s=7)
    if goal_radius is not None:
        goal_circle = Circle(goal_state[[0, 2]], goal_radius, color='g',
                             alpha=0.3)
        ax.add_patch(goal_circle)
    # visualize goal
    if rrt.goal_node is not None:
        plot_state = rrt.goal_node.nominal_state
        ax.scatter([plot_state[0]], [plot_state[2]], c='c', s=7)

    # visualize nodes and edges
    node_queue = deque()
    node_queue.append(rrt.root_node)

    tree_path_segments = []
    while(len(node_queue) > 0) and show_all_nodes:
        # pop from the top of the queue
        node = node_queue.popleft()
        # get the children of this node and add to the queue
        children_nodes = node.children
        node_queue.extend(children_nodes)
        # visualize the state
        plot_state = node.nominal_state
        ax.scatter([plot_state[0]], [plot_state[2]], c='grey', s=1.)
        # Visualize the randup state randomly
        # randup_state_count = node.states_cluster.shape[0]
        # random_vis_state_indices = np.random.choice(
        #     np.arange(randup_state_count),
        #     min(randup_state_count, 10),
        #     replace=False)
        # for ri in random_vis_state_indices:
        #     rs = node.states_cluster[ri,:]
        #     ax.scatter([rs[0]], [rs[1]], c='grey', alpha=0.4, s=0.5)
    #     if visualize_all_paths and node.parent is not None and node.path_from_parent is not None:
    #         #SLOW!
    #         #reconstruct dubin's path on the fly
    #         segs = node.path_from_parent.get_dubins_interpolated_path()
    #         for i in range(segs.shape[0]-1):
        if node.parent is not None:
            tree_path_segments.append(
                [node.nominal_state[[0, 2]], node.parent.nominal_state[[0, 2]]])
    if show_all_nodes:
        tree_lc = mc.LineCollection(
            tree_path_segments, colors='k', linewidths=0.2, alpha=0.5)
        ax.add_collection(tree_lc)

    # Plot the best node
    color = 'turquoise'
    # if rrt.goal_node is not None:
    #     color = 'turquoise'
    # else:
    #     color = 'y'
    node = rrt.best_node
    goal_path_segments = []
    # Green goal node
    # segs = node.path_from_parent.get_dubins_interpolated_path()
    # for i in range(segs.shape[0] - 1):
    #     goal_path_segments.append([segs[i, 0:2], segs[i + 1, 0:2]])
    plot_state = node.nominal_state
    ax.scatter([plot_state[0]], [plot_state[2]], c=color, s=2.)

    # plot the convex hull
    plot_convex_hull(
        node.states_cluster[:, [0, 2]], ax, color, alpha=.75, lw=0.25)
    parent_plot_state = node.parent.nominal_state
    # Connect the states
    # ax.plot([plot_state[0], parent_plot_state[0]],
    #         [plot_state[2], parent_plot_state[2]],
    #         'b-', alpha=0.7, linewidth=0.7)
    node = node.parent
    while node.parent != None:
        # segs = node.path_from_parent.get_dubins_interpolated_path()
        # for i in range(segs.shape[0] - 1):
        #     goal_path_segments.append([segs[i, 0:2], segs[i + 1, 0:2]])
        plot_state = node.nominal_state
        ax.scatter([plot_state[0]], [plot_state[2]], c='b', s=2.)
        parent_plot_state = node.parent.nominal_state
        # Connect the quivers
        # ax.plot([plot_state[0], parent_plot_state[0]],
        #         [plot_state[2], parent_plot_state[2]],
        #         'b-', alpha=0.7, linewidth=0.7)
        plot_convex_hull(
            node.states_cluster[:, [0, 2]], ax, 'b', alpha=.75, lw=0.25)
        node = node.parent
    # Plot the root
    plot_state = node.nominal_state
    ax.scatter([plot_state[0]], [plot_state[2]], c='r', s=7.)

    # Plot the obstacles:
    if world is not None:
        world.visualize(ax)
    # if world_type is not None:
    #      for obstacle_tuple in world_type.obstacle_tuples:
    #          if type(obstacle_tuple) == tuple:
    #              obstacle_pose, obstacle_type = obstacle_tuple
    #          else:
    #              obstacle_pose = obstacle_tuple
    #              obstacle_type = ObstacleType.NORMAL_BOX
    #          R = umath.get_R_AB(obstacle_pose[2])
    #          apices = []
    #          for coeff in [(1,1), (-1, 1), (-1, -1), (1, -1)]:
    #              if obstacle_type == ObstacleType.NORMAL_BOX:
    #                  half_length = param.BOX_HALF_LENGTH
    #                  half_width = param.BOX_HALF_WIDTH
    #              elif obstacle_type == ObstacleType.LONG_BOX_6X:
    #                  half_length = param.LONG_BOX_HALF_LENGTH
    #                  half_width = param.LONG_BOX_HALF_WIDTH
    #              else:
    #                  raise NotImplementedError
    #              apex = R @ np.asarray([half_length*coeff[0],
    #                                     half_width*coeff[1]])+\
    #                     obstacle_pose[:2]
    #              apices.append(apex)
    #          apices = np.asarray(apices)
    #          polygon = Polygon(apices, color='gray')
    #          ax.add_patch(polygon)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin=-2,xmax=16)
    ax.set_ylim(ymin=-1,ymax=7)
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')

    # # Label the obstacles
    plt.text(1.3,5.1, "$\mathcal{X}_G$")
    plt.text(-0.4, 0.3, "$x_0$")
    plt.text(8.2, 0.4, "$\mathcal{C}$")
    plt.text(7, -0.9, "$\mathcal{C}$")
    plt.text(-0.9, 2.8, "$\mathcal{C}$")
    plt.text(14.2, 5.1, "$\mathcal{C}$")
    plt.text(13.2, 1.5, "$\mathcal{C}$")

    return fig, ax

