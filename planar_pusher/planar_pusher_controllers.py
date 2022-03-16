import utils.umath as umath
import planar_pusher.param as param
import numpy as np
import scipy.optimize as opt
import warnings


class ProjectionPDNLPController:
    def __init__(self, friction_cone_half_aperature,
                 gains, xytheta_weighting,
                 f_max, mu_g):
        self.psi = friction_cone_half_aperature
        self.cone_coeff_matrix = \
            np.asarray([[np.cos(self.psi), np.cos(self.psi)],
                        [np.sin(self.psi), -np.sin(self.psi)]])
        self.cone_coeff_inv = np.linalg.pinv(self.cone_coeff_matrix)
        self.gains = gains
        self.xytheta_weighting = xytheta_weighting
        self.f_max = f_max
        self.optimization_bounds = np.asarray([[0., 1.],
                                               [-self.psi,
                                                self.psi]])
        # self.last_ans = np.average(self.optimization_bounds, axis=1)
        self.mu_g = mu_g
        warnings.warn("This controller is deprecated")

    def compute_action(self, current_state, desired_state):
        state_diff = desired_state-current_state
        pd = state_diff*self.gains
        state_dd_des = pd[:3]+pd[3:]
        if np.linalg.norm(current_state[3:5]) > 1e-1:
            friction_comp = self.mu_g*0.5
        else:
            friction_comp = 0.
        v_theta = np.arctan2(current_state[4], current_state[3])

        def loss(arg):
            f, phi = arg
            f *= self.f_max
            xdd = -f / param.BOX_MASS * \
                np.cos(
                    phi + current_state[param.BOX_STATE_THETA]) - np.cos(v_theta) * friction_comp
            ydd = -f / param.BOX_MASS *\
                np.sin(
                    phi + current_state[param.BOX_STATE_THETA]) - np.sin(v_theta) * friction_comp
            thetadd = -f * param.BOX_HALF_LENGTH /\
                param.BOX_INERTIA * np.sin(phi)
            state_dd = np.array([xdd, ydd, thetadd])
            return np.sum((state_dd_des-state_dd)**2)
        res = opt.minimize(loss, np.random.uniform(self.optimization_bounds[:, 0],
                                                   self.optimization_bounds[:, 1]),
                           bounds=self.optimization_bounds)
        if np.linalg.norm(current_state[3:5]) <= 1e-1 and res.x[0] > 1e-1:
            res.x[0] += self.mu_g
        f_B = np.asarray([np.cos(res.x[1]), -np.sin(res.x[1])])*res.x[0]
        basis_coeff = self.cone_coeff_inv @ f_B
        # phi_W = current_state[param.BOX_STATE_THETA]+res.x[1]
        # f_W_p = umath.get_R_AB(phi_W)[0:]*res.x[0]
        # print('fwp', f_W_p)
        # b_array = np.vstack([b0_W, b1_W]).T
        # b_inv = np.linalg.pinv(b_array)
        # # Project the state difference onto the friction cone
        # basis_coeff = b_inv @ -f_W_p
        # # For testing purposes
        # print('basis', basis_coeff)
        return np.array([0., *np.maximum(basis_coeff, 0.)])


class ProjectionController:
    def __init__(self, half_friction_cone_aperture,
                 gains, f_max, mu_g):
        self.psi = half_friction_cone_aperture
        self.cone_coeff_matrix = \
            np.asarray([[np.cos(self.psi), np.cos(self.psi)],
                        [np.sin(self.psi), -np.sin(self.psi)]])
        self.cone_coeff_inv = np.linalg.pinv(self.cone_coeff_matrix)
        self.gains = gains
        self.f_max = f_max
        self.theta_gain_max = np.inf
        self.optimization_bounds = np.asarray([[-1., -1.],
                                               [self.f_max, self.f_max]])
        # self.last_ans = np.average(self.optimization_bounds, axis=1)
        self.mu_g = mu_g

    def compute_action(self, current_state, desired_state):
        state_diff = desired_state-current_state
        state_diff[param.BOX_STATE_THETA] = umath.angle_diff_wrapped(
            desired_state[param.BOX_STATE_THETA],
            current_state[param.BOX_STATE_THETA]
        )
        R_W_B = umath.get_R_AB(current_state[param.BOX_STATE_THETA])
        state_diff_B = np.copy(state_diff)
        # Rotate the position and velocity so it's w.r.t. forward direction
        # of the box
        state_diff_B[:2] = R_W_B.T@state_diff_B[:2]
        state_diff_B[3:5] = R_W_B.T@state_diff_B[3:5]

        pd = state_diff_B*self.gains
        state_dd_des = pd[:3]+pd[3:]
        # res = opt.lsq_linear(-self.cone_coeff_matrix, state_dd_des[:2],
        #                      bounds=self.optimization_bounds)
        # coeffs = res.x
        coeffs = -self.cone_coeff_inv@state_dd_des[:2]
        coeffs = np.maximum(np.minimum(coeffs, self.f_max), 0.)
        # # Do PD on theta
        position_diff = np.linalg.norm(state_diff_B[:2])
        if state_dd_des[param.BOX_STATE_THETA] > 1e-2:
            coeffs[0] += min(abs(state_dd_des[2]),
                             self.theta_gain_max)*min(1., position_diff)
        elif state_dd_des[param.BOX_STATE_THETA] < -1e-2:
            coeffs[1] += min(abs(state_dd_des[2]),
                             self.theta_gain_max)*min(1., position_diff)
        return np.array([0., *np.maximum(coeffs, 0.)])
