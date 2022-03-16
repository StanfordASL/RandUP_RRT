import planar_pusher.param as param
import utils.umath as umath
import numpy as np
import pybullet as p
from enum import Enum
import warnings
import os


class Object:
    def __init__(self, object_file, bullet_client,
                 np_random=None, **kwargs):
        self.object_file = object_file

        self.np_random = np_random
        # To be set in reset()
        self._p = bullet_client  # bullet session to connect to
        self.p_W_init = None
        self.q_W_init = None
        self.body_id = None
        # self.body_link_id = None
        # Friction coefficient
        self.mu = None
        self.half_friction_cone_aperture = None

    def reset_object(self):
        raise NotImplementedError

    def set_state(self, **kwargs):
        raise NotImplementedError

    def print_all_joints_info(self):
        raise NotImplementedError

    def apply_action(self, actions):
        raise NotImplementedError

    def get_object_observation(self):
        raise NotImplementedError

    def get_current_p_B_W_and_q_B_W(self):
        raise NotImplementedError

    def _compute_dynamics_properties(self):
        raise NotImplementedError

    def compute_p_W_and_n_W(self, phi_B):
        """
        Given a planar angle in the rigid_object frame B, perform ray cast from the
        origin of the B frame and find the corresponding surface normal and
        ray impact position in the world frame W.
        FIXME(wualbert): We don't care about z as we are in 2D.
                         This may cause problems later.
        @param phi_B: The yaw of the contact point in the B frame.
        @return: (p_W, n_W). p_W is the 3D contact point on the rigid_object.
        n_W is the 3D outward-facing unit normal vector at the contact point.
        Both of these are in the world frame.
        """
        # Get the pose of the rigid_object frame origin in world frame
        p_B_W, q_B_W = self.get_current_p_B_W_and_q_B_W()
        e_B_W = self._p.getEulerFromQuaternion(q_B_W)
        # add phi_B
        e_phi_W = e_B_W + np.asarray([0., 0., phi_B])
        e_phi_W %= 2*np.pi
        q_phi_W = self._p.getQuaternionFromEuler(e_phi_W)
        # Construct the unit ray vector
        # Compute the ray to cast
        ray_direction_W = np.array(
            self._p.getMatrixFromQuaternion(q_phi_W)).reshape(3, 3)[:, 0]
        # FIXME(wualbert): the ray shouldn't be of arbitrary length
        ray_start_W = p_B_W + ray_direction_W*10.
        ray_end_W = p_B_W - ray_direction_W * 10.
        ans = self._p.rayTest(ray_start_W, ray_end_W)
        for a in ans:
            if a[0] == self.body_id:
                # FIXME(wualbert): a cleaner way to do this
                # We only care about the normal of this particular rigid_object
                return a[3:]
        raise AssertionError

    def compute_friction_cone(self, phi_B):
        """
        Compute the bases of the friction cone.
        @param phi_B: The yaw of the contact point in the B frame.
        @return: (p_W, b0_W, b1_W). p_W is the 3D position of the apex.
        b0_W and b1_W are 3D unit basis vectors spanning the friction cone.
        By convention, the yaw increases from b0_W to b1_W.
        Since we are in 2D, the z's of all 3 vectors should be identical.
        """
        cos = np.cos(self.half_friction_cone_aperture)
        sin = np.sin(self.half_friction_cone_aperture)
        p_W, n_W = self.compute_p_W_and_n_W(phi_B)
        assert(cos >= 0)
        assert(sin >= 0)
        R = np.zeros((3, 3))
        R[:2, :2] = np.asarray([[cos, -sin], [sin, cos]])
        b0_W = np.dot(R.T, n_W)
        b1_W = np.dot(R, n_W)
        assert(b0_W[-1] == 0)
        assert (b1_W[-1] == 0)
        return p_W, b0_W, b1_W

    def compute_f_W_p(self, phi_B, alpha_0, alpha_1):
        """
        Compute the force applied on the rigid_object as ᵂfᵖ=-α₀⋅ᵂb₀-α₁⋅ᵂb₁
        The force is pushing into the rigid_object, hence the negation
        @param phi_B: The yaw of the contact point in the B frame.
        @param alpha_0: coefficient on b0_W
        @param alpha_1: coefficient on b1_W
        @return: (p_W, f_W_p), p_W is the 3D contact point on the rigid_object.
        f_W_p is the force applied
        """
        assert alpha_0 >= 0.
        assert alpha_1 >= 0.
        # First compute the force vectors
        p_W, b0_W, b1_W = self.compute_friction_cone(phi_B)
        # Compute the force
        f_W_p = -alpha_0*b0_W-alpha_1*b1_W
        assert f_W_p[-1] == 0
        return np.array(p_W), np.array(f_W_p)


class BoxObject2D(Object):
    def __init__(self, bullet_client, mu=0.5, **kwargs):
        self.body_dof = 6  # [x, y, theta] and derivatives
        self.body_link_id = param.BOX_2D_BODY_LINK  # Specified in model file
        self._p = bullet_client
        # texture
        if 'path' not in kwargs:
            self.path = os.path.dirname(
                os.path.realpath(__file__))+"/assets/2x1x05_2d.sdf"
        else:
            self.path = kwargs['path']
        self.texture_id = self._p.loadTexture(os.path.dirname(
            os.path.realpath(__file__))+"/assets/arrow_blue.png")
        super().__init__(self.path, bullet_client, **kwargs)
        self.mu = mu
        self.reset_object()
        # For visualization and debugging
        self.action_visualization_ids = []
        self.external_force_visualization_ids = []

    def _anchor_base_and_allow_2D_movement(self):
        if self.body_id < 0:
            # In case the rigid_object hasn't been initialized yet
            return
        # Anchor the rigid_object at the origin except z = param.BOX_2D_HALF_HEIGHT
        self._p.resetBasePositionAndOrientation(self.body_id,
                                                [0., 0., param.BOX_2D_HALF_HEIGHT],
                                                [0., 0., 0., 1.])
        self.base_anchor_constraint_id = self._p.createConstraint(self.body_id, -1, -1, -1, self._p.JOINT_FIXED, [1, 1, 1], [0, 0, 0],
                                                                  [0., 0., param.BOX_2D_HALF_HEIGHT])
        maxForce = 0
        mode = self._p.VELOCITY_CONTROL
        for j in range(self._p.getNumJoints(self.body_id)):
            self._p.enableJointForceTorqueSensor(self.body_id, j)
            self._p.setJointMotorControl2(self.body_id, j,
                                          controlMode=mode, force=maxForce)

    def compute_p_W_and_n_W(self, phi_B, use_raytrace=False):
        """
        Given a planar angle in the rigid_object frame B, perform ray cast from the
        origin of the B frame and find the corresponding surface normal and
        ray impact position in the world frame W.
        FIXME(wualbert): We don't care about z as we are in 2D.
                         This may cause problems later.
        @param phi_B: The yaw of the contact point in the B frame.
        @return: (p_W, n_W). p_W is the 3D contact point on the rigid_object.
        n_W is the 3D outward-facing unit normal vector at the contact point.
        Both of these are in the world frame.
        """
        # Get the pose of the rigid_object frame origin in world frame
        if phi_B % (np.pi/2) == 0 and not use_raytrace:
            # FIXME(wualbert): raytracing seems to be slower than
            #                   get_current_p_B_W_and_q_B_W() by one step, so
            #                   the closed form answer may be slightly different
            return self.compute_p_W_and_n_special(phi_B)
        warnings.warn("Using ray tracing to compute p_W and n_W, "
                      "which is not compatible with obstacles")
        p_B_W, q_B_W = self.get_current_p_B_W_and_q_B_W()
        e_B_W = self._p.getEulerFromQuaternion(q_B_W)
        # add phi_B
        e_phi_W = e_B_W + np.asarray([0., 0., phi_B])
        e_phi_W %= 2*np.pi
        q_phi_W = self._p.getQuaternionFromEuler(e_phi_W)
        # Construct the unit ray vector
        # Compute the ray to cast
        ray_direction_W = np.array(
            self._p.getMatrixFromQuaternion(q_phi_W)).reshape(3, 3)[:, 0]
        # FIXME(wualbert): the ray shouldn't be of arbitrary length
        ray_start_W = p_B_W + ray_direction_W*3.
        ray_end_W = p_B_W - ray_direction_W * 3.
        ans = self._p.rayTest(ray_start_W, ray_end_W)
        for a in ans:
            if a[0] == self.body_id:
                # FIXME(wualbert): a cleaner way to do this
                # We only care about the normal of this particular rigid_object
                return a[3:]
        raise AssertionError

    def compute_p_W_and_n_special(self, phi_B):
        p_B_W, q_B_W = self.get_current_p_B_W_and_q_B_W()
        e_B_W = self._p.getEulerFromQuaternion(q_B_W)
        R = umath.get_R_AB(e_B_W[2])
        # compute contact point location in world
        phi_special = phi_B % (2*np.pi)
        if phi_special == 0:
            p_W = R[:, param.BOX_STATE_X] * \
                param.BOX_HALF_LENGTH+np.array(p_B_W[:2])
            n_W = R[:, param.BOX_STATE_X]
        elif phi_special == np.pi/2.:
            p_W = R[:, param.BOX_STATE_Y] * \
                param.BOX_HALF_WIDTH+np.array(p_B_W[:2])
            n_W = R[:, param.BOX_STATE_Y]
        elif phi_special == np.pi:
            p_W = -R[:, param.BOX_STATE_X] * \
                param.BOX_HALF_LENGTH+np.array(p_B_W[:2])
            n_W = -R[:, param.BOX_STATE_X]
        elif phi_special == np.pi*3./2.:
            p_W = -R[:, param.BOX_STATE_Y] * \
                param.BOX_HALF_WIDTH+np.array(p_B_W[:2])
            n_W = -R[:, param.BOX_STATE_Y]
        else:
            raise NotImplementedError
        p_W = np.hstack([p_W, np.array([param.BOX_2D_HALF_HEIGHT])])
        n_W = np.hstack([n_W, np.zeros(1)])
        return p_W, n_W

    def get_current_p_B_W_and_q_B_W(self):
        ans = self._p.getLinkState(
            self.body_id, self.body_link_id, computeLinkVelocity=True)
        return ans[0], ans[1]

    def reset_object(self):
        if self.object_file.split('.')[-1] == "sdf":
            self.body_id = self._p.loadSDF(self.object_file)[0]
        elif self.object_file.split('.')[-1] == "urdf":
            self.body_id = self._p.loadURDF(self.object_file)
        else:
            raise NotImplementedError
        self._p.changeDynamics(
            self.body_id, self.body_link_id, lateralFriction=self.mu)
        self._compute_dynamics_properties()
        self._anchor_base_and_allow_2D_movement()
        self._p.changeVisualShape(
            self.body_id, self.body_link_id, textureUniqueId=self.texture_id)

    def set_state(self, state):
        '''
        Set the state of the rigid_object
        :param state:
        :return:
        '''
        assert(len(state) == self.body_dof)
        self._p.resetJointState(
            self.body_id, param.BOX_2D_X_LINK, state[param.BOX_STATE_X],
            state[param.BOX_STATE_X_DOT])
        self._p.resetJointState(
            self.body_id, param.BOX_2D_Y_LINK, state[param.BOX_STATE_Y],
            state[param.BOX_STATE_Y_DOT])
        self._p.resetJointState(
            self.body_id, self.body_link_id, state[param.BOX_STATE_THETA],
            state[param.BOX_STATE_THETA_DOT])

    def print_all_joints_info(self):
        raise NotImplementedError

    def _compute_dynamics_properties(self):
        np.testing.assert_allclose(self.mu, self._p.getDynamicsInfo(
            self.body_id, self.body_link_id)[1])
        self.half_friction_cone_aperture = np.arctan(self.mu)

    def apply_action(self, action):
        """
        An action is a three tuple (ϕ, α₀, α₁)
        :param actions: list of action from all agents
        :return:
        """
        warnings.warn(
            "Using deprecated BoxObject2D. Friction is broken with this.")
        # clear old IDs
        assert len(action) == 3
        # FIXME(wualbert): clip action space in a better way
        p_W, f_W_p = self.compute_f_W_p(*action)
        self._p.applyExternalForce(
            self.body_id, self.body_link_id, f_W_p, p_W, self._p.WORLD_FRAME)
        if self._p.getConnectionInfo()['connectionMethod'] == p.GUI:
            for id in self.action_visualization_ids:
                self._p.removeUserDebugItem(id)
            self.action_visualization_ids.clear()
            self.action_visualization_ids.append(
                self._p.addUserDebugLine(p_W-param.FORCE_VIS_SCALING*f_W_p, p_W,
                                         lineColorRGB=param.FORCE_VIS_COLOR[0]))

    def apply_external_f_tau_on_body(self, f_W_p=None, tau_W_p=None):
        """
        :return:
        """
        p_B_W, q_B_W = self.get_current_p_B_W_and_q_B_W()
        if f_W_p is not None:
            self._p.applyExternalForce(
                self.body_id, self.body_link_id, f_W_p, p_B_W, self._p.WORLD_FRAME)
        if tau_W_p is not None:
            # In 2D, x and y torques are always zero
            np.testing.assert_allclose(tau_W_p[:2], np.zeros(2))
            self._p.applyExternalTorque(
                self.body_id, self.body_link_id, tau_W_p, self._p.WORLD_FRAME)

    def get_object_observation(self):
        """

        :return: [x, y, θ, ẋ, ẏ, \dot{θ}̇]
        """
        obs = []
        ans = self._p.getLinkState(self.body_id, self.body_link_id,
                                   computeLinkVelocity=True)
        # Only care about [x, y, θ]
        # x, y
        obs.extend(ans[0][:2])
        # θ
        yaw = self._p.getEulerFromQuaternion(ans[1])[2]
        obs.append(yaw)
        # x, y derivatives
        obs.extend(ans[6][:2])
        # θ derivative
        obs.append(ans[7][2])
        return np.asarray(obs)


class GoalObject:
    def __init__(self, bullet_client, pose):
        self._p = bullet_client
        self.pose = pose
        self.p_W = np.zeros(3)
        self.p_W[:2] = pose[:2]
        self.p_W[2] = param.BOX_2D_HALF_HEIGHT
        self.q_W = self._p.getQuaternionFromEuler(
            np.asarray([0., 0., pose[2]]))
        self.object_file = os.path.dirname(
            os.path.realpath(__file__))+"/assets/2x1x05_vis.sdf"
        self.reset_object()

    def reset_object(self):
        if self.object_file.split('.')[-1] == "sdf":
            self.body_id = self._p.loadSDF(self.object_file)[0]
        elif self.object_file.split('.')[-1] == "urdf":
            self.body_id = self._p.loadURDF(self.object_file)
        else:
            raise NotImplementedError
        # Set pose
        self._p.resetBasePositionAndOrientation(self.body_id, self.p_W,
                                                self.q_W)
        # Change color to green
        self.texture_id = self._p.loadTexture(os.path.dirname(
            os.path.realpath(__file__))+"/assets/arrow_green.png")
        self._p.changeVisualShape(
            self.body_id, -1, textureUniqueId=self.texture_id)


class ObstacleType(Enum):
    NORMAL_BOX = 0
    LONG_BOX_6X = 1


class Obstacle(BoxObject2D):
    def __init__(self, bullet_client, pose,
                 type=ObstacleType.NORMAL_BOX):
        self.pose = pose
        self.q_B_W = p.getQuaternionFromEuler(
            [0., 0., self.pose[param.BOX_STATE_THETA]])
        self.type = type
        if type == ObstacleType.NORMAL_BOX:
            path = os.path.dirname(
                os.path.realpath(__file__))+"/assets/2x1x05_2d.sdf"
        elif type == ObstacleType.LONG_BOX_6X:
            path = os.path.dirname(
                os.path.realpath(__file__)) + "/assets/12x1x05_2d.sdf"
        else:
            raise NotImplementedError
        super().__init__(bullet_client=bullet_client, path=path)
        assert(pose.shape == (3,))

    def reset_object(self):
        super().reset_object()
        self.texture_id = self._p.loadTexture(os.path.dirname(
            os.path.realpath(__file__))+"/assets/arrow_white.png")
        # self._p.changeVisualShape(
        #     self.body_id, self.body_link_id, textureUniqueId=self.texture_id)
        # Anchor to pose
        self._anchor_to_pose()

    def _anchor_to_pose(self):
        self._p.resetJointState(
            self.body_id, param.BOX_2D_X_LINK, self.pose[param.BOX_STATE_X], 0)
        self._p.resetJointState(
            self.body_id, param.BOX_2D_Y_LINK, self.pose[param.BOX_STATE_Y], 0)
        self._p.resetJointState(
            self.body_id, self.body_link_id, self.pose[param.BOX_STATE_THETA], 0)
        self.pose_constraint = self._p.createConstraint(
            self.body_id,
            self.body_link_id, -1,
            -1, self._p.JOINT_FIXED, [1, 1, 1], [0, 0, 0],
            [self.pose[param.BOX_STATE_X], self.pose[param.BOX_STATE_Y],
             param.BOX_2D_HALF_HEIGHT], [
                0., 0., 0., 1.],
            self.q_B_W)

    def apply_action(self, *args):
        """
        Obstacle should not get action
        :param action:
        :return:
        """
        raise AttributeError

    def set_state(self, *args):
        """
        Obstacle should not get state set
        :param action:
        :return:
        """
        raise AttributeError


class BoxObject(Object):
    def __init__(self, bullet_client, mu=0.5, **kwargs):
        self.body_dof = 6  # [x, y, theta] and derivatives
        self._p = bullet_client
        # texture
        if 'path' not in kwargs:
            self.path = os.path.dirname(
                os.path.realpath(__file__))+"/assets/2x1x05.sdf"
        else:
            self.path = kwargs['path']
        self.texture_id = self._p.loadTexture(os.path.dirname(
            os.path.realpath(__file__))+"/assets/arrow_red.png")
        super().__init__(self.path, bullet_client, **kwargs)
        self.body_link_id = -1  # Specified in model file
        self.mu = mu
        self.reset_object()
        # For visualization and debugging
        self.action_visualization_ids = []
        self.external_force_visualization_ids = []

    def reset_object(self):
        if self.object_file.split('.')[-1] == "sdf":
            self.body_id = self._p.loadSDF(self.object_file)[0]
        elif self.object_file.split('.')[-1] == "urdf":
            self.body_id = self._p.loadURDF(self.object_file)
        else:
            raise NotImplementedError
        self._p.changeDynamics(
            self.body_id, self.body_link_id, lateralFriction=self.mu)
        self._compute_dynamics_properties()
        self._p.changeVisualShape(
            self.body_id, self.body_link_id, textureUniqueId=self.texture_id)
        # Reset to origin
        self._p.resetBasePositionAndOrientation(self.body_id,
                                                [0., 0., param.BOX_2D_HALF_HEIGHT],
                                                [0., 0., 0., 1.])

    def get_current_p_B_W_and_q_B_W(self):
        p_W, q_W = self._p.getBasePositionAndOrientation(self.body_id)
        return p_W, q_W

    def _compute_dynamics_properties(self):
        np.testing.assert_allclose(self.mu, self._p.getDynamicsInfo(
            self.body_id, self.body_link_id)[1])
        self.half_friction_cone_aperture = np.arctan(self.mu)

    def apply_external_f_tau_on_body(self, f_W_p=None, tau_W_p=None):
        """
        :return:
        """
        p_B_W, q_B_W = self.get_current_p_B_W_and_q_B_W()
        if f_W_p is not None:
            self._p.applyExternalForce(
                self.body_id, self.body_link_id, f_W_p, p_B_W, self._p.WORLD_FRAME)
        if tau_W_p is not None:
            # In 2D, x and y torques are always zero
            np.testing.assert_allclose(tau_W_p[:2], np.zeros(2))
            self._p.applyExternalTorque(
                self.body_id, self.body_link_id, tau_W_p, self._p.WORLD_FRAME)

        p_B_W, _ = self.get_current_p_B_W_and_q_B_W()
        p_B_W = np.asarray(p_B_W)
        p_B_W[2] = param.BOX_2D_HALF_HEIGHT*2

        if self._p.getConnectionInfo()['connectionMethod'] == p.GUI:
            # Visualize external force
            for id in self.external_force_visualization_ids:
                self._p.removeUserDebugItem(id)
            self.external_force_visualization_ids.clear()
            self.external_force_visualization_ids.append(
                self._p.addUserDebugLine(p_B_W-param.DISTURBANCE_VIS_SCALING*f_W_p, p_B_W,
                                         lineColorRGB=param.DISTURBANCE_VIS_COLOR))

    def get_object_observation(self):
        """

        :return: [x, y, θ, ẋ, ẏ, \dot{θ}̇]
        """
        obs = np.zeros(6)
        p_B_W, q_B_W = self.get_current_p_B_W_and_q_B_W()
        p_B_W_dot, rpy_dot = self._p.getBaseVelocity(self.body_id)
        # Only care about [x, y, θ]
        # x, y
        obs[:param.BOX_STATE_THETA] = p_B_W[:param.BOX_STATE_THETA]
        # θ
        obs[param.BOX_STATE_THETA] = self._p.getEulerFromQuaternion(q_B_W)[2]
        # x, y derivatives
        obs[param.BOX_STATE_THETA+1:param.BOX_STATE_THETA_DOT] = p_B_W_dot[:2]
        # θ derivative
        obs[param.BOX_STATE_THETA_DOT] = rpy_dot[2]
        return np.asarray(obs)

    def set_state(self, state):
        '''
        Set the state of the rigid_object
        :param state:
        :return:
        '''
        assert(len(state) == self.body_dof)
        p_B_W_desired = [state[param.BOX_STATE_X],
                         state[param.BOX_STATE_Y], param.BOX_2D_HALF_HEIGHT]
        q_B_W_desired = self._p.getQuaternionFromEuler(
            [0., 0., state[param.BOX_STATE_THETA]])
        self._p.resetBasePositionAndOrientation(self.body_id,
                                                p_B_W_desired, q_B_W_desired)
        p_B_W_dot_desired = [state[param.BOX_STATE_X+3],
                             state[param.BOX_STATE_Y+3], 0.]
        rpy_dot = [0., 0., state[param.BOX_STATE_THETA+3]]
        self._p.resetBaseVelocity(self.body_id, p_B_W_dot_desired,
                                  rpy_dot)

    def apply_action(self, action):
        """
        An action is a three tuple (ϕ, α₀, α₁)
        :param actions: list of action from all agents
        :return:
        """
        # clear old IDs
        assert len(action) == 3
        # FIXME(wualbert): clip action space in a better way
        p_W, f_W_p = self.compute_f_W_p(*action)
        self._p.applyExternalForce(
            self.body_id, self.body_link_id, f_W_p, p_W, self._p.WORLD_FRAME)
        if self._p.getConnectionInfo()['connectionMethod'] == p.GUI:
            for id in self.action_visualization_ids:
                self._p.removeUserDebugItem(id)
            self.action_visualization_ids.clear()
            self.action_visualization_ids.append(
                self._p.addUserDebugLine(p_W-param.FORCE_VIS_SCALING*f_W_p, p_W,
                                         lineColorRGB=param.FORCE_VIS_COLOR[0]))

    def compute_p_W_and_n_W(self, phi_B, use_raytrace=False):
        """
        Given a planar angle in the rigid_object frame B, perform ray cast from the
        origin of the B frame and find the corresponding surface normal and
        ray impact position in the world frame W.
        FIXME(wualbert): We don't care about z as we are in 2D.
                         This may cause problems later.
        @param phi_B: The yaw of the contact point in the B frame.
        @return: (p_W, n_W). p_W is the 3D contact point on the rigid_object.
        n_W is the 3D outward-facing unit normal vector at the contact point.
        Both of these are in the world frame.
        """
        # Get the pose of the rigid_object frame origin in world frame
        if phi_B % (np.pi/2) == 0 and not use_raytrace:
            # FIXME(wualbert): raytracing seems to be slower than
            #                   get_current_p_B_W_and_q_B_W() by one step, so
            #                   the closed form answer may be slightly different
            return self.compute_p_W_and_n_special(phi_B)
        warnings.warn("Using ray tracing to compute p_W and n_W, "
                      "which is not compatible with obstacles")
        p_B_W, q_B_W = self.get_current_p_B_W_and_q_B_W()
        e_B_W = self._p.getEulerFromQuaternion(q_B_W)
        # add phi_B
        e_phi_W = e_B_W + np.asarray([0., 0., phi_B])
        e_phi_W %= 2*np.pi
        q_phi_W = self._p.getQuaternionFromEuler(e_phi_W)
        # Construct the unit ray vector
        # Compute the ray to cast
        ray_direction_W = np.array(
            self._p.getMatrixFromQuaternion(q_phi_W)).reshape(3, 3)[:, 0]
        # FIXME(wualbert): the ray shouldn't be of arbitrary length
        ray_start_W = p_B_W + ray_direction_W*3.
        ray_end_W = p_B_W - ray_direction_W * 3.
        ans = self._p.rayTest(ray_start_W, ray_end_W)
        for a in ans:
            if a[0] == self.body_id:
                # FIXME(wualbert): a cleaner way to do this
                # We only care about the normal of this particular rigid_object
                return a[3:]
        raise AssertionError

    def compute_p_W_and_n_special(self, phi_B):
        p_B_W, q_B_W = self.get_current_p_B_W_and_q_B_W()
        e_B_W = self._p.getEulerFromQuaternion(q_B_W)
        R = umath.get_R_AB(e_B_W[2])
        # compute contact point location in world
        phi_special = phi_B % (2*np.pi)
        if phi_special == 0:
            p_W = R[:, param.BOX_STATE_X] * \
                param.BOX_HALF_LENGTH+np.array(p_B_W[:2])
            n_W = R[:, param.BOX_STATE_X]
        elif phi_special == np.pi/2.:
            p_W = R[:, param.BOX_STATE_Y] * \
                param.BOX_HALF_WIDTH+np.array(p_B_W[:2])
            n_W = R[:, param.BOX_STATE_Y]
        elif phi_special == np.pi:
            p_W = -R[:, param.BOX_STATE_X] * \
                param.BOX_HALF_LENGTH+np.array(p_B_W[:2])
            n_W = -R[:, param.BOX_STATE_X]
        elif phi_special == np.pi*3./2.:
            p_W = -R[:, param.BOX_STATE_Y] * \
                param.BOX_HALF_WIDTH+np.array(p_B_W[:2])
            n_W = -R[:, param.BOX_STATE_Y]
        else:
            raise NotImplementedError
        p_W = np.hstack([p_W, np.array([param.BOX_2D_HALF_HEIGHT])])
        n_W = np.hstack([n_W, np.zeros(1)])
        return p_W, n_W

# class SphericalPusher(Object):
#     def __init__(self, p_W_init=None, q_W_init=None, action_sapce=None,
#                  **kwargs):
#         if p_W_init is None:
#             p_W_init = np.asarray([0., 0., param.BOX_2D_HALF_HEIGHT])
#         else:
#             assert(p_W_init[2] == param.BOX_2D_HALF_HEIGHT)
#         if q_W_init is None:
#             q_W_init = np.asarray([0., 0., 0., 1.])
#         super().__init__(os.path.dirname(
#             os.path.realpath(__file__))+"/assets/finger_ball.sdf", p_W_init,
#             q_W_init, **kwargs)
#         self.body_dof = 3  # [x, y, theta]
#         self.body_link_id = NotImplementedError
#         self.action_space = action_sapce
#         # Change texture
#         self.texture_id = self._p.loadTexture(os.path.dirname(
#             os.path.realpath(__file__))+"/assets/arrow_green.png")
#         print('texture', self.texture_id)
#         # For visualization and debugging
#         self.action_visualization_ids = []
#         if self._p is not None:
#             self.reset(self._p)
#
#     def get_current_p_B_W_and_q_B_W(self):
#         ans = self._p.getLinkState(self.body_id, self.body_link_id, computeLinkVelocity=True)
#         return ans[0], ans[1]
#
#     def _anchor_object_and_allow_movement(self, p_W, q_W):
#         if self.body_id < 0:
#             # In case the rigid_object hasn't been initialized yet
#             return
#         # FIXME(wualbert): should we anchor at the rigid_object origin?
#         self._p.resetBasePositionAndOrientation(self.body_id, p_W, q_W)
#         self.base_anchor_constraint_id = self._p.createConstraint(self.body_id, -1, -1, -1, self._p.JOINT_FIXED, [1, 1, 1], [0, 0, 0],
#                                                 p_W)
#         maxForce = 0
#         mode = self._p.VELOCITY_CONTROL
#         for j in range(self._p.getNumJoints(self.body_id)):
#             self._p.enableJointForceTorqueSensor(self.body_id, j)
#             self._p.setJointMotorControl2(self.body_id, j,
#                                     controlMode=mode, force=maxForce)
#
#     def reset(self, bullet_client):
#         self._p = bullet_client
#         if self.object_file.split('.')[-1] == "sdf":
#             self.body_id = self._p.loadSDF(self.object_file)[0]
#         elif self.object_file.split('.')[-1] == "urdf":
#             self.body_id = self._p.loadURDF(self.object_file)
#         else:
#             raise NotImplementedError
#
#         p_W = self.p_W_init
#         self._compute_dynamics_properties()
#         self._anchor_object_and_allow_movement(p_W, self.q_W_init)
#         self._p.changeVisualShape(self.body_id, 0, textureUniqueId=self.texture_id)
#
#     def get_object_observation(self):
#         """
#
#         :return: List [3D p_W, 4D q_W, 3D p_dot_W, 3D angular velocity]
#         """
#         obs = []
#         ans = self._p.getLinkState(self.body_id, self.body_link_id, computeLinkVelocity=True)
#         # Only care about [x, y, θ]
#         # x, y
#         obs.extend(ans[0][:2])
#         # θ
#         yaw = self._p.getEulerFromQuaternion(ans[1])[2]
#         obs.append(yaw)
#         # x, y derivatives
#         obs.extend(ans[6][:2])
#         # θ derivative
#         obs.append(ans[7][2])
#         return obs
