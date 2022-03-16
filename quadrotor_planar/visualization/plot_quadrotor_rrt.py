import utils.umath as umath
from collections import deque
from enum import Enum
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Circle
from planar_pusher.visualization.plot_planar_pusher_rrt import plot_convex_hull
import numpy as np
import scipy.spatial
import warnings
import scipy.spatial.qhull
import time

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 24})

def visualize_quadrotor_rrt(rrt, goal_state=None, show_all_nodes=False,
                            show_all_arrows=False,
                            ax=None, fig=None, visualize_all_paths=True,
                            obstacle_list=None,
                            padding=0.,
                            goal_threshold=0.):
    def get_plotting_nominal_state(states_cluster):
        plotting_cluster = np.copy(states_cluster)
        plotting_nominal_state = np.average(plotting_cluster, axis=0)
        return plotting_nominal_state

    if ax is None or fig is None:
        fig, ax = plt.subplots()
    # visualize start
    plot_state = get_plotting_nominal_state(rrt.root_node.states_cluster)
    # ax.quiver([plot_state[0]], [plot_state[1]],
    #           [plot_state[2]], [plot_state[3]], facecolor='r',
    #           scale=.2, scale_units='dots', width=0.01, pivot='mid')
    ax.scatter([plot_state[0]], [plot_state[1]], c='r', alpha=1., s=3.)
    if goal_state is not None:
        # ax.quiver([goal_state[0]],
        #           [goal_state[1]],
        #           [goal_state[2]],
        #           [goal_state[3]], facecolor='g',
        #           scale=.2, scale_units='dots', width=0.01, pivot='mid')
        ax.scatter([goal_state[0]], [goal_state[1]], c='g', alpha=1., s=3.)
        # plot the goal region
        cir = Circle(goal_state, goal_threshold, color='g', alpha=0.2)
        ax.add_patch(cir)
        cir = Circle(goal_state, (goal_threshold-padding),
                     color='g', alpha=0.4)
        ax.add_patch(cir)
    # visualize goal
    if rrt.goal_node is not None:
        plot_state = get_plotting_nominal_state(rrt.goal_node.states_cluster)
        # ax.quiver([plot_state[0]], [plot_state[1]],
        #           [plot_state[2]], [plot_state[3]], facecolor='c',
        #           scale=.2, scale_units='dots', width=0.01, pivot='mid')
        print("Goal node", plot_state)
        ax.scatter([plot_state[0]], [plot_state[1]], c='c', alpha=1., s=5.)
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
        if show_all_arrows:
            ax.quiver([plot_state[0]], [plot_state[1]],
                      [plot_state[2]],
                      [plot_state[3]], facecolor='grey',
                      scale=.2, scale_units='dots', width=0.002, pivot='mid', alpha=0.4)
        else:
            ax.quiver([plot_state[0]], [plot_state[1]],
                      [plot_state[2]],
                      [plot_state[3]], facecolor='grey',
                      scale=1000, scale_units='dots', width=0.002, pivot='mid', alpha=0.4)
        # Visualize the randup state randomly
        # randup_state_count = node.states_cluster.shape[0]
        # random_vis_state_indices = np.random.choice(
        #     np.arange(randup_state_count),
        #     min(randup_state_count, 10),
        #     replace=False)
        # for ri in random_vis_state_indices:
        #     rs = node.states_cluster[ri, :]
        #     ax.scatter([rs[0]], [rs[1]], c='grey', alpha=0.4, s=0.5)
        if visualize_all_paths and node.parent is not None:
            # SLOW!
            # reconstruct dubin's path on the fly
            parent_plot_state = get_plotting_nominal_state(
                node.parent.states_cluster)
            tree_path_segments.append((parent_plot_state[:2], plot_state[:2]))
    if visualize_all_paths:
        tree_lc = mc.LineCollection(
            tree_path_segments, colors='k', linewidths=0.2, alpha=0.3)
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
    plot_state = get_plotting_nominal_state(node.states_cluster)
    # ax.quiver([plot_state[0]], [plot_state[1]],
    #           [plot_state[2]], [plot_state[3]], facecolor=color,
    #           scale=.2, scale_units='dots', width=0.002, pivot='mid', alpha=1)
    ax.scatter([plot_state[0]], [plot_state[1]], c=color, alpha=1., s=3.)
    # plot the convex hull
    plot_convex_hull(node.states_cluster[:, :2], ax, color, alpha=0.5, lw=0.25)
    parent_plot_state = get_plotting_nominal_state(node.parent.states_cluster)
    # Connect the quivers
    ax.plot([plot_state[0], parent_plot_state[0]],
            [plot_state[1], parent_plot_state[1]],
            'b-', alpha=0.7, linewidth=1.)
    node = node.parent
    while node.parent != None:
        # segs = node.path_from_parent.get_dubins_interpolated_path()
        # for i in range(segs.shape[0] - 1):
        #     goal_path_segments.append([segs[i, 0:2], segs[i + 1, 0:2]])
        plot_state = get_plotting_nominal_state(node.states_cluster)
        # ax.quiver([plot_state[0]], [plot_state[1]],
        #           [plot_state[2]], [plot_state[3]], facecolor='b',
        #           scale=.2, scale_units='dots', width=0.002, pivot='mid', alpha=1)
        ax.scatter([plot_state[0]], [plot_state[1]], c='b', alpha=1., s=1.)
        parent_plot_state = get_plotting_nominal_state(
            node.parent.states_cluster)
        # Connect the quivers
        ax.plot([plot_state[0], parent_plot_state[0]],
                [plot_state[1], parent_plot_state[1]],
                'b-', alpha=0.7, linewidth=0.5)
        plot_convex_hull(
            node.states_cluster[:, :2], ax, 'b', alpha=0.75, lw=0.25)
        node = node.parent
    # Plot the root
    plot_state = get_plotting_nominal_state(node.states_cluster)
    # ax.quiver([plot_state[0]], [plot_state[1]],
    #           [plot_state[2]], [plot_state[3]], facecolor='r',
    #           scale=.2, scale_units='dots', width=0.002, pivot='mid', alpha=1)
    ax.scatter([plot_state[0]], [plot_state[1]], c='r', alpha=1., s=5.)

    # Plot the obstacles
    if obstacle_list is not None:
        for center, radius in obstacle_list:
            cir = Circle(center, radius, color='gray', alpha=0.4)
            ax.add_patch(cir)
            cir = Circle(center, radius+padding, color='gray', alpha=0.1)
            ax.add_patch(cir)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin=-1,xmax=11.1)
    ax.set_ylim(ymin=-2,ymax=2)
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    return fig, ax
