"""
src/lever_arm.py

Applies a physical lever-arm offset between the Vive Tracker origin
and the EV3 kinematic center.
"""

import numpy as np

def apply_lever_arm_2d(vive_x: np.ndarray, vive_y: np.ndarray, vive_theta: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the tracker positions and the tracker's heading (theta),
    subtract the lever-arm offset (dx, dy) in the local frame to find
    the robot's kinematic center.

    x_robot = x_vive - (dx * cos(theta) - dy * sin(theta))
    y_robot = y_vive - (dx * sin(theta) + dy * cos(theta))
    """
    if dx == 0.0 and dy == 0.0:
        return vive_x, vive_y

    cos_t = np.cos(vive_theta)
    sin_t = np.sin(vive_theta)

    robot_x = vive_x - (dx * cos_t - dy * sin_t)
    robot_y = vive_y - (dx * sin_t + dy * cos_t)

    return robot_x, robot_y
