from enum import Enum


class ContinuousSystemModes(Enum):
    MODE = 0


class DriftingCarSystemModes(Enum):
    GRIP = 0
    SLIP = 1


class HybridIntegratorModes(Enum):
    CONTACT = 0
    FLIGHT = 1
