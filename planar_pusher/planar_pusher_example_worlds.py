import planar_pusher.planar_pusher_world as planar_pusher_world
from dataclasses import dataclass
from planar_pusher.rigid_object import ObstacleType
import numpy as np
from pybullet_utils import bullet_client


@dataclass
class FreePlanarPusherWorld:
    obstacle_poses: np.ndarray = np.asarray([])
    floor_friction: float = 1.
    time_step: float = 1. / 240.
    number_of_steps: int = 120


@dataclass
class PlanarPusherControlTuningWorld:
    obstacle_tuples: np.ndarray = np.asarray([])
    floor_friction: float = 1.
    time_step: float = 1. / 90.
    number_of_steps: int = 45


@dataclass
class ObstacleTestWorld:
    obstacle_poses = np.asarray([[2.2, 0., 0.], [0., 1.7, np.pi / 2]])
    floor_friction: float = 1.
    time_step: float = 1. / 240.
    number_of_steps: int = 24


@dataclass
class SimpleObstacleWorld:
    obstacle_poses = np.asarray([[-4, 0, np.pi/2]])
    floor_friction: float = 1.
    time_step: float = 1. / 120.
    number_of_steps: int = 60


@dataclass
class MediumObstacleWorld:
    obstacle_poses = np.asarray(
        [[-4, 0, np.pi/2], [0., 3., 0.], [-1., -2.5, np.pi/3]])
    floor_friction: float = 1.
    time_step: float = 1. / 120.
    number_of_steps: int = 60


@dataclass
class NarrowPassageWorld:
    gap = 2.0
    low_obs = -gap/2.-1
    high_obs = gap/2.+1
    obstacle_tuples = [
        (np.array([-5, low_obs, np.pi/2]), ObstacleType.NORMAL_BOX),
        (np.array([-5, low_obs-2, np.pi / 2]), ObstacleType.NORMAL_BOX),
        (np.array([-5, high_obs+5., np.pi / 2]), ObstacleType.LONG_BOX_6X)
    ]
    floor_friction: float = 1.2
    action_time_step: float = 1. / 60.
    number_of_steps: int = 60
    time_step: float = 1. / 90.
    number_of_steps: int = 90


def create_planar_pusher_world(world, **kwargs):
    return planar_pusher_world.BoxManipWorld(world.floor_friction,
                                             world.action_time_step,
                                             number_of_steps=world.number_of_steps,
                                             obstacle_tuples=world.obstacle_tuples,
                                             **kwargs)
