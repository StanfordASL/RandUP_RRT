import planar_pusher.param as param
import utils.umath as umath
import planar_pusher.planar_pusher_example_worlds as planar_pusher_example_worlds
from planar_pusher.rigid_object import ObstacleType
from collections import deque
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.patches import Circle
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
    p = Polygon(states[hull_vertices, :2], *args, **kwargs)
    ax.add_patch(p)


base_apices = []
for s1, s2 in [(1,1),(1,-1),(-1,-1),(-1,1)]:
    base_apices.append([s1*param.BOX_HALF_LENGTH, s2*param.BOX_HALF_WIDTH])
base_apices = np.array(base_apices).T

def get_box_polygon(state, color, alpha):
    theta = state[param.OBJ_ORIENT_START]
    R = np.asarray([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    apices = state[:param.OBJ_ORIENT_START]+(R@base_apices).T
    polygon = Polygon(apices, color=color, alpha=alpha)
    return polygon

def visualize_planar_pusher_continuous_rrt(rrt, goal_state=None,
                                           show_all_nodes=False,
                                           ax=None, fig=None,
                                           world_type=None,
                                           goal_threshold=0.):
    def get_plotting_nominal_state(states_cluster):
        plotting_cluster = np.copy(states_cluster)
        plotting_cluster[:, rrt.wrap_indices][
            plotting_cluster[:, rrt.wrap_indices] > np.pi] -= 2 * np.pi
        plotting_nominal_state = np.average(plotting_cluster, axis=0)
        return plotting_nominal_state

    if ax is None or fig is None:
        fig, ax = plt.subplots()
    # visualize start
    plot_state = get_plotting_nominal_state(rrt.root_node.states_cluster)
    ax.quiver([plot_state[0]], [plot_state[1]],
              [np.cos(plot_state[2])], [np.sin(plot_state[2])], facecolor='r',
              scale=20, width=0.002, pivot='mid')
    ax.add_patch(get_box_polygon(plot_state, 'r', 0.2))
    if goal_state is not None:
        ax.quiver([goal_state[0]],
                  [goal_state[1]],
                  [np.cos(goal_state[2])],
                  [np.sin(goal_state[2])], facecolor='g',
                  scale=20, width=0.007, pivot='mid')
        ax.add_patch(get_box_polygon(goal_state, 'g', 0.2))
    # visualize goal
    if rrt.goal_node is not None:
        plot_state = get_plotting_nominal_state(rrt.goal_node.states_cluster)
        ax.quiver([plot_state[0]], [plot_state[1]],
                  [np.cos(plot_state[2])], [np.sin(plot_state[2])], facecolor='c',
                  scale=20, width=0.007, pivot='mid')
        ax.add_patch(get_box_polygon(plot_state, 'c', 0.2))
    cir = Circle(goal_state, goal_threshold, color='g', alpha=0.2)
    ax.add_patch(cir)
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
        plot_state = get_plotting_nominal_state(node.states_cluster)
        ax.quiver([plot_state[0]], [plot_state[1]],
                  [np.cos(plot_state[2])],
                  [np.sin(plot_state[2])], facecolor='grey',
                  scale=40, width=0.002, pivot='mid', alpha=0.4)
        # Visualize the randup state randomly
        randup_state_count = node.states_cluster.shape[0]
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
    #             tree_path_segments.append([segs[i,0:2], segs[i+1,0:2]])
    # if visualize_all_paths:
    #     tree_lc = mc.LineCollection(tree_path_segments, colors='k', linewidths=0.2, alpha=0.5)
    #     ax.add_collection(tree_lc)

    # Plot the best node
    color = 'b'
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
    plot_state = get_plotting_nominal_state(node.states_cluster)
    ax.quiver([plot_state[0]], [plot_state[1]],
              [np.cos(plot_state[2])], [np.sin(plot_state[2])], facecolor=color,
              scale=20, width=0.007, pivot='mid', alpha=1)
    # Best node
    ax.add_patch(get_box_polygon(plot_state, color, 0.2))
    # plot the convex hull
    # plot_convex_hull(node.states_cluster[:, :2], ax, color, alpha=0.5, lw=0.25)
    parent_plot_state = get_plotting_nominal_state(node.parent.states_cluster)
    # Connect the quivers
    ax.plot([plot_state[0], parent_plot_state[0]],
            [plot_state[1], parent_plot_state[1]],
            'b-', alpha=0.7, linewidth=0.7)
    node = node.parent
    while node.parent != None:
        # segs = node.path_from_parent.get_dubins_interpolated_path()
        # for i in range(segs.shape[0] - 1):
        #     goal_path_segments.append([segs[i, 0:2], segs[i + 1, 0:2]])
        plot_state = get_plotting_nominal_state(node.states_cluster)
        ax.quiver([plot_state[0]], [plot_state[1]],
                  [np.cos(plot_state[2])], [np.sin(plot_state[2])], facecolor='b',
                  scale=20, width=0.007, pivot='mid', alpha=1)
        ax.add_patch(get_box_polygon(plot_state, 'b', 0.2))
        parent_plot_state = get_plotting_nominal_state(
            node.parent.states_cluster)
        # Connect the quivers
        ax.plot([plot_state[0], parent_plot_state[0]],
                [plot_state[1], parent_plot_state[1]],
                'b-', alpha=0.7, linewidth=0.7)
        # plot_convex_hull(
        #     node.states_cluster[:, :2], ax, 'b', alpha=0.7, lw=0.25)
        node = node.parent
    # Plot the root
    plot_state = get_plotting_nominal_state(node.states_cluster)
    ax.quiver([plot_state[0]], [plot_state[1]],
              [np.cos(plot_state[2])], [np.sin(plot_state[2])], facecolor='r',
              scale=20, width=0.007, pivot='mid', alpha=1)
    # Plot the pusher worlds:
    if world_type is not None:
        for obstacle_tuple in world_type.obstacle_tuples:
            if type(obstacle_tuple) == tuple:
                obstacle_pose, obstacle_type = obstacle_tuple
            else:
                obstacle_pose = obstacle_tuple
                obstacle_type = ObstacleType.NORMAL_BOX
            R = umath.get_R_AB(obstacle_pose[2])
            apices = []
            for coeff in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
                if obstacle_type == ObstacleType.NORMAL_BOX:
                    half_length = param.BOX_HALF_LENGTH
                    half_width = param.BOX_HALF_WIDTH
                elif obstacle_type == ObstacleType.LONG_BOX_6X:
                    half_length = param.LONG_BOX_HALF_LENGTH
                    half_width = param.LONG_BOX_HALF_WIDTH
                else:
                    raise NotImplementedError
                apex = R @ np.asarray([half_length*coeff[0],
                                       half_width*coeff[1]]) +\
                    obstacle_pose[:2]
                apices.append(apex)
            apices = np.asarray(apices)
            polygon = Polygon(apices, color='gray')
            ax.add_patch(polygon)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin=-19,xmax=7)
    ax.set_ylim(ymin=-10,ymax=15)
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    # Label the obstacles
    plt.text(-17,1.5, "$\mathcal{X}_G$")
    plt.text(3, 1, "$x_0$")
    plt.text(-7.5, 6, "$\mathcal{C}$")
    plt.text(-7.5, -4, "$\mathcal{C}$")
    return fig, ax

def visualize_planar_pusher_trajectory(trajectory,
                                       fig=None,ax=None,
                                       color='b',
                                       alpha=0.5
                                       ):
    plot_state = trajectory[0]
    # ax.quiver([plot_state[0]], [plot_state[1]],
    #           [np.cos(plot_state[2])], [np.sin(plot_state[2])], facecolor=color,
    #           scale=20, width=0.005, pivot='mid', alpha=alpha)
    for idx in range(1,len(trajectory)):
        plot_state = trajectory[idx]
        parent_plot_state = trajectory[idx-1]
        ax.plot([plot_state[0], parent_plot_state[0]],
                [plot_state[1], parent_plot_state[1]],
                color=color, linestyle='-', alpha=0.5, linewidth=1.)
        ax.add_patch(get_box_polygon(plot_state, color, alpha))
    return fig, ax