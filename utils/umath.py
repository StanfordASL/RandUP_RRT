import numpy as np


def get_R_AB(thetaB_A):
    '''
    Get the rotation matrix from a frame B to a frame A. p_A = R_AB*p_B
    @param thetaB_A: the rotation of B with respect to A
    @return: 2*2 numpy array R_AB.
    '''
    cos_theta = np.cos(thetaB_A)
    sin_theta = np.sin(thetaB_A)
    return np.asarray([[cos_theta, -sin_theta],
                       [sin_theta, cos_theta]])


def angle_diff_wrapped(theta_1, theta_2):
    '''
    return theta_1 - theta_2 but "wrapped around"
    i.e. answer with the smallest absolute value
    The answer is between -pi~pi
    :param theta_1:
    :param theta_2:
    :return:
    '''
    diff_raw = theta_1-theta_2
    diff = np.arctan2(
        np.sin(diff_raw), np.cos(diff_raw)
    )
    return diff


def scaled_diff(x1, x2, scaling_array):
    return scaling_array*(x1-x2)


def scaled_norm(x1, x2, scaling_array):
    return np.linalg.norm(scaled_diff(x1, x2, scaling_array))
