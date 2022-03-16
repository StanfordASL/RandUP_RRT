import time

import planar_pusher.param as param
import utils.umath as umath
import planar_pusher.rigid_object as rigid_object
import planning.contact_modes as contact_modes

from pybullet_utils import bullet_client
import pybullet as p
import pybullet_data

import numpy as np

import os
import inspect
import warnings

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))


class BoxManipWorld:
    def __init__(self,
                 floor_friction,
                 time_step,
                 np_random=None,
                 number_of_steps=24,
                 bullet_p=None,
                 obstacle_tuples=None,
                 goal_pose=None,
                 **kwargs
                 ):
        self.time_step_size = time_step
        # kwargs
        if bullet_p is not None:
            self._p = bullet_p
            if "use_gui" in kwargs:
                warnings.warn(
                    "Bullet Client already provided. use_gui not used")
        else:
            if "use_gui" in kwargs:
                self.use_gui = kwargs["use_gui"]
            else:
                self.use_gui = False
            warnings.warn("Bullet Client Not Supplied")
            if self.use_gui:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient()
        if np_random is None:
            np_random = np.random.RandomState()
        self.np_random = np_random
        # TODO(wualbert): support multiple robots

        self.objects = [rigid_object.BoxObject(self._p)]
        self.box_object_index = 0
        if obstacle_tuples is not None:
            if len(obstacle_tuples) == 0:
                pass
            elif type(obstacle_tuples[0]) == tuple:
                for obstacle_tuple in obstacle_tuples:
                    obstacle = rigid_object.Obstacle(self._p, obstacle_tuple[0],
                                                     obstacle_tuple[1])
                    self.objects.append(obstacle)
            else:
                warnings.warn("Deprecated. Obstacle type not specified.")
                for obstacle in obstacle_tuples:
                    obstacle = rigid_object.Obstacle(self._p, obstacle)
                    self.objects.append(obstacle)

        self.goal_pose = goal_pose
        if self.goal_pose is not None:
            self.goal_object = rigid_object.GoalObject(self._p, self.goal_pose)
            self.objects.append(self.goal_object)

        # used once temporarily, will be overwritten outside though superclass api
        self.floor_id = None

        self.floor_friction = floor_friction
        self.number_of_steps = number_of_steps
        obs = self.reset()    # and update init obs

    def reset(self):
        print('reset env')
        self._p.resetSimulation()
        self._p.setTimeStep(self.time_step_size)
        self._p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # optionally
        self.floor_id = self._p.loadURDF("plane.urdf")

        self._p.changeDynamics(self.floor_id, -1,
                               lateralFriction=self.floor_friction)
        self._p.setGravity(0., 0., param.GRAVITY)
        for obj in self.objects:
            obj.reset_object()
        # Step simulation so ray tracing works
        self._p.stepSimulation()
        # change the perturbation maximum
        obs = self.get_state()
        return np.array(obs)

    def set_state(self, state, no_step=True):
        """
        :param state:
        :param no_step:
        :return:
        """
        assert(len(state) == 6)
        self.objects[self.box_object_index].set_state(state)
        if not no_step:
            # Step simulation so ray tracing works
            self._p.stepSimulation()

    #
    # def step(self, a_list):
    #     """Run one timestep of the environment's dynamics. When end of
    #     episode is reached, you are responsible for calling `reset()`
    #     to reset this environment's state.
    #
    #     Accepts an list of actions and returns a tuple (observation, reward, done, info).
    #
    #     Args:
    #         action (rigid_object): an list of actions, each provided by the agent
    #
    #     Returns:
    #         observation (rigid_object): agent's observation of the current environment
    #         reward (float) : amount of reward returned after previous action
    #         done (bool): whether the episode has ended, in which case further step() calls will return undefined results
    #         info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    #     """
    #     info = dict()
    #     for _ in range(self.control_skip):
    #         a_list = np.atleast_2d(a_list)
    #         if self.clip_action_values:
    #             mapped_action = np.asarray(list(self.map_nn_output_to_action(a) for a in a_list))
    #         else:
    #             mapped_action = a_list
    #         self.objects[self.box_object_index].apply_action(mapped_action)
    #         # The perturbation generated on the last time step is used.
    #         # This way the policy can see the perturbation before deciding the action
    #         if self.perturb_f_tau is not None:
    #             self.objects[self.box_object_index].apply_external_f_tau_on_body(*self.perturb_f_tau)
    #         self._p.stepSimulation()
    #     # prepare the next perturbation
    #     if self.perturb_f_tau_fn is not None:
    #         self.perturb_f_tau = self.perturb_f_tau_fn(self.perturb_size)
    #     done = False
    #     obs = self.get_observation()
    #     # Calculate reward
    #     # TODO(wualbert): better implementation
    #     reward, done, extra_info = self._compute_reward(obs)
    #     info.update(extra_info)
    #     return obs, reward, done, info

    def simulate_forward_and_check_collision(self, starting_state, local_controller,
                                             noise_magnitude,
                                             noise_update_hold=30,
                                             desired_mode=None,
                                             override_step_size=None,
                                             override_number_of_steps=None,
                                             change_simulator_state=False,
                                             collision_check_interval=None,
                                             no_step=True,
                                             sleep=False):
        """

        :param starting_state:
        :param local_controller:
        :param noise_magnitude:
        :param desired_mode:
        :param override_step_size:
        :param override_number_of_steps:
        :param change_simulator_state:
        :param check_collision:
        :param no_step: whether to step simulation after set_state
        :return:
        """
        assert(starting_state.shape == (6,))
        cached_state = self.get_state()
        self.set_state(starting_state, no_step)
        has_collision = False
        number_of_steps = self.number_of_steps
        if override_step_size is not None:
            # self._p.setTimeStep(override_step_size)
            raise NotImplementedError
        if override_number_of_steps is not None:
            number_of_steps = override_number_of_steps
        if collision_check_interval is None:
            collision_check_interval = 10
        f_W_p = np.zeros(3)
        for step in range(number_of_steps):
            # Compute control action
            action = local_controller(self.get_state())
            self.objects[self.box_object_index].apply_action(action)
            # Apply the noise
            if noise_magnitude is not None:
                if step % noise_update_hold == 0:
                    random_angle = self.np_random.uniform(0., 2 * np.pi)
                    random_magnitude = self.np_random.uniform(
                        0., noise_magnitude)
                    f_W_p = np.zeros(3)
                    f_W_p[:2] = umath.get_R_AB(random_angle) @ np.asarray(
                        [random_magnitude, 0])
                self.objects[
                    self.box_object_index].apply_external_f_tau_on_body(
                    f_W_p=f_W_p)
            # TODO(wualbert): add noise
            self._p.stepSimulation()
            if (step+1) % collision_check_interval == 0:
                ans = self._p.getContactPoints(bodyA=self.objects[self.box_object_index].body_id,
                                               linkIndexA=self.objects[self.box_object_index].body_link_id)
                obstacle_collisions = []
                for a in ans:
                    if a[1] == self.floor_id or a[2] == self.floor_id:
                        continue
                    elif a[1] == self.objects[self.box_object_index].body_id or \
                            a[2] == self.objects[self.box_object_index].body_id:
                        obstacle_collisions.append(a)
                has_collision = len(obstacle_collisions) > 0 or has_collision
            if has_collision:
                break
        ending_state = self.get_state()
        ending_state[param.BOX_STATE_THETA] %= 2*np.pi
        if not change_simulator_state:
            self.set_state(cached_state, no_step)
        if override_step_size is not None:
            # self._p.setTimeStep(self.time_step_size)
            raise NotImplementedError
        resulting_mode = contact_modes.ContinuousSystemModes.MODE
        # TODO(wualbert): implementation for different modes
        return ending_state, resulting_mode, has_collision

    def compute_f_tau_on_body_from_action(self, action):
        """
        Helper function for getting the world-frame action (the type used by
        apply_external_f_tau_on_body()) from the parameterized action (the one
        used by step())
        :param a: parameterized action
        :return: (f_W_p, tau_B), the resulting force and torque on the body
        center.
        """
        p_f_W, f_W_p = self.objects[self.box_object_index].compute_f_W_p(
            *action)
        p_W = np.array(
            self.objects[self.box_object_index].get_current_p_B_W_and_q_B_W()[0])
        # tau = r x f
        tau_B = np.cross(p_f_W-p_W, f_W_p)
        return f_W_p, tau_B

    def get_state(self):
        """
        The observation is flattened into a 1D array. It is calculated
        with respect to the goal
        :return:
        """
        obs = self.objects[self.box_object_index].get_object_observation()
        return np.asarray(obs)

# class BoxFingerManipEnv2D(gym.Env):
#     raise NotImplementedError
