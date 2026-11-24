"""Geometry for the official RoboMini Gazebo scene, in metres and radians."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


AXIS = np.array([0.0, 0.928])  # World X/Z coordinates, shared by both faces.
RADIUS = 0.350
LAMP_OFFSET = 0.0475
TARGET_STEP = 2.0 * math.pi / 5
PLATE_FRONT = 0.127
LAMP_FRONT = 0.128
GRAVITY = 9.81
PITCH_JOINT_Z = 0.145  # SDF joint pose is relative to the pitch_link.
CAMERA_MOUNT = np.array([0., 0.036, 0.075 - PITCH_JOINT_Z])
MUZZLE_LENGTH = 0.095
SPAWN_LENGTH = MUZZLE_LENGTH + 0.025
OPTICAL_TO_LINK = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
LAMP_POINTS = np.array([[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]]) * LAMP_OFFSET


def rotation(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([[cy, -sy * cp, sy * sp], [sy, cy * cp, -cy * sp], [0., sp, cp]])


@dataclass(frozen=True)
class Scene:
    side: str = 'blue'

    def __post_init__(self):
        if self.side not in ('blue', 'red'):
            raise ValueError('side must be blue or red')

    @property
    def sign(self):
        return 1 if self.side == 'blue' else -1

    @property
    def pivot(self):
        return np.array([0., -2.07 * self.sign, 0.495 + PITCH_JOINT_Z])

    def link_rotation(self, yaw, pitch):
        return rotation(yaw + (math.pi if self.sign < 0 else 0.), pitch)

    def target(self, angle, depth=PLATE_FRONT):
        return np.array([RADIUS * math.cos(angle), -depth * self.sign, AXIS[1] + RADIUS * math.sin(angle)])

    def barrel(self, yaw, pitch, spawn=False):
        mount = self.link_rotation(yaw, pitch)
        origin = self.pivot + mount @ [0., SPAWN_LENGTH if spawn else MUZZLE_LENGTH, -PITCH_JOINT_Z]
        return origin, mount[:, 1]

    def angles_to(self, target, yaw, pitch):
        origin, _ = self.barrel(yaw, pitch)
        dx, dy, dz = target - origin
        target_yaw = math.atan2(-self.sign * dx, self.sign * dy)
        target_pitch = math.atan2(dz, math.hypot(dx, dy))
        return math.remainder(target_yaw - yaw, 2.0 * math.pi), target_pitch - pitch


@dataclass(frozen=True)
class Intrinsics:
    matrix: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True)
class Camera:
    intrinsics: Intrinsics
    origin: np.ndarray
    optical_to_world: np.ndarray

    @classmethod
    def on_gimbal(cls, scene, intrinsics, yaw, pitch):
        mount = scene.link_rotation(yaw, pitch)
        return cls(intrinsics, scene.pivot + mount @ CAMERA_MOUNT, mount @ OPTICAL_TO_LINK)

    def to_optical(self, points):
        return (np.asarray(points) - self.origin) @ self.optical_to_world

    def project(self, points):
        optical = self.to_optical(points).reshape(-1, 3)
        pixels, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), self.intrinsics.matrix, self.intrinsics.distortion)
        return pixels.reshape(-1, 2)

    def intersect_lamp_plane(self, pixels, scene):
        xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), self.intrinsics.matrix, self.intrinsics.distortion)
        rays = np.column_stack((xy.reshape(-1, 2), np.ones(len(pixels)))) @ self.optical_to_world.T
        forward = scene.sign * rays[:, 1] > 0.05
        scale = (-scene.sign * LAMP_FRONT - self.origin[1]) / np.where(forward, rays[:, 1], np.nan)
        return self.origin + rays * scale[:, None]


def plate_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [0., 0., -1.], [s, c, 0.]])


def flight_solution(scene, target, speed):
    """Low arc from the rotated spawn point to the target."""
    dx, dy, dz = target - scene.pivot
    horizontal = math.hypot(dx, dy)
    yaw = math.atan2(-scene.sign * dx, scene.sign * dy)
    pitch = math.atan2(dz + PITCH_JOINT_Z, horizontal)
    for _ in range(6):
        cosine = math.cos(pitch)
        flight = ((horizontal - PITCH_JOINT_Z * math.sin(pitch)) / cosine - SPAWN_LENGTH) / speed
        pitch = math.atan2(dz + PITCH_JOINT_Z / cosine + 0.5 * GRAVITY * flight ** 2, horizontal)
    flight = ((horizontal - PITCH_JOINT_Z * math.sin(pitch)) / math.cos(pitch) - SPAWN_LENGTH) / speed
    return yaw, pitch, flight


def impact(scene, yaw, pitch, speed):
    origin, direction = scene.barrel(yaw, pitch, spawn=True)
    if scene.sign * direction[1] <= 0.05:
        return None
    flight = (-scene.sign * PLATE_FRONT - origin[1]) / (speed * direction[1])
    point = origin + speed * flight * direction
    point[2] -= 0.5 * GRAVITY * flight ** 2
    return point, flight
