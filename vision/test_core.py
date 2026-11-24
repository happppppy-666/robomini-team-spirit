"""Offline checks for geometry and lamp recognition, using generated frames."""

from dataclasses import dataclass
import math
import time
import unittest

import cv2
import numpy as np

from robomini_vision.detector import detect
from robomini_vision.geometry import (
    Camera, Intrinsics, LAMP_FRONT, LAMP_POINTS, Scene, TARGET_STEP,
    flight_solution, impact, plate_rotation,
)
from robomini_vision.tracking import WheelTracker


@dataclass(frozen=True)
class Observation:
    angle: float
    state: str


K = Intrinsics(np.array([[1108.5, 0., 640.], [0., 1108.5, 360.], [0., 0., 1.]]), np.zeros(5))


def make_frame(scene, camera, phase, current, completed):
    image = np.full((720, 1280, 3), (120, 127, 130), dtype=np.uint8)
    color = (248, 68, 9) if scene.side == 'blue' else (12, 27, 254)
    for index in range(5):
        angle = phase + index * TARGET_STEP
        center = scene.target(angle, LAMP_FRONT)
        transform = plate_rotation(angle)
        u, v = np.rint(camera.project([center])[0]).astype(int)
        cv2.circle(image, (u, v), 49, (32, 33, 35), -1)
        for lamp, point in enumerate(LAMP_POINTS):
            a, b = (0.0205, 0.010) if lamp < 2 else (0.010, 0.0205)
            corners = point + np.array([[-a, -b, 0.], [a, -b, 0.], [a, b, 0.], [-a, b, 0.]])
            polygon = np.rint(camera.project(corners @ transform.T + center)).astype(np.int32)
            lit = index == current or (index in completed and lamp >= 2)
            cv2.fillConvexPoly(image, polygon, color if lit else (19, 20, 21))
    return image


class VisionCoreTest(unittest.TestCase):
    def test_lamp_states_pose_and_angles(self):
        cv2.setNumThreads(2)
        worst_world = worst_rms = worst_camera_m = 0.
        count = 0
        start = time.perf_counter()
        for side in ('red', 'blue'):
            scene = Scene(side)
            for index in range(60):
                yaw = 0.08 * math.cos(index / 12)
                pitch = 0.21 + 0.06 * math.sin(index / 12)
                phase = index * 0.097
                current = index % 5
                completed = [(current + j) % 5 for j in range(1, index % 5 + 1)]
                camera = Camera.on_gimbal(scene, K, yaw, pitch)
                sample = make_frame(scene, camera, phase, current, completed)
                observed = detect(sample, camera, scene)
                self.assertEqual(len(observed), len(completed) + 1)
                self.assertEqual([m.state for m in observed].count('pending'), 1)
                target = next(m for m in observed if m.state == 'pending')
                count += 1
                expected_angle = phase + current * TARGET_STEP
                true_center = scene.target(expected_angle)
                worst_world = max(worst_world, float(np.linalg.norm(target.world_center - true_center)))
                worst_rms = max(worst_rms, target.reprojection_error)
                xyz = camera.to_optical(scene.target(expected_angle, LAMP_FRONT))
                worst_camera_m = max(worst_camera_m, float(np.linalg.norm(target.translation - xyz)))
                delta_yaw, delta_pitch = scene.angles_to(target.world_center, yaw, pitch)
                _, direction = scene.barrel(yaw + delta_yaw, pitch + delta_pitch)
                origin, _ = scene.barrel(yaw, pitch)
                ray = target.world_center - origin
                cos_error = direction @ ray / np.linalg.norm(ray)
                self.assertGreater(cos_error, 1. - 1e-8)
        self.assertLess(worst_world, 0.004)
        self.assertLess(worst_camera_m, 0.08)
        self.assertLess(worst_rms, 1.)
        print(f'{count} synthetic frames, max world error {worst_world*1000:.2f} mm, '
              f'PnP optical error {worst_camera_m*1000:.2f} mm, max RMS {worst_rms:.2f} px '
              f'({time.perf_counter()-start:.2f} s)')

    def test_flight_solution_matches_impact(self):
        worst = 0.
        for side in ('red', 'blue'):
            scene = Scene(side)
            for phase in np.linspace(-math.pi, math.pi, 81):
                for speed in (9., 12., 15.):
                    target = scene.target(float(phase))
                    yaw, pitch, _ = flight_solution(scene, target, speed)
                    point, duration = impact(scene, yaw, pitch, speed)
                    worst = max(worst, float(np.linalg.norm(point - target)))
                    self.assertGreater(duration, 0.)
        self.assertLess(worst, 0.00001)
        print(f'486 flight solutions, maximum analytical impact error {worst*1000:.4f} mm')

    def test_rotor_motion_prediction_across_handoffs(self):
        def motions(t):
            return (0., -math.pi / 3 * t,
                    -(2.09 - 0.96) * t + (0.96 / 1.955) * math.cos(1.955 * t + 0.31))
        rng = np.random.default_rng(114)
        worst = [0.] * 3
        for mode in range(3):
            tracker = WheelTracker()
            for index in range(240):
                t = index / 60
                phase = motions(t)[mode]
                current = int(t / 0.38) % 5
                pending = phase + current * TARGET_STEP
                items = [Observation(pending + rng.normal(0., 0.001), 'pending'),
                         Observation(pending + TARGET_STEP + rng.normal(0., 0.001), 'active')]
                estimate = tracker.observe(t, items)
                if t > 1.35:
                    actual = motions(t + 0.18)[mode] - phase
                    worst[mode] = max(worst[mode], abs(estimate.offset(t, t + 0.18) - actual))
        self.assertLess(max(worst), 0.045)
        print('180 ms rotor prediction max error (rad), static/small/big:', [round(v, 4) for v in worst])


if __name__ == '__main__':
    unittest.main()
