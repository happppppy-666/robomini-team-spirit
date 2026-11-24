"""Classify lamp patterns in the camera image and estimate the pending plate pose."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .geometry import AXIS, LAMP_FRONT, LAMP_OFFSET, LAMP_POINTS, RADIUS, TARGET_STEP, plate_rotation


TANGENT_ANGLE = math.atan2(LAMP_OFFSET, RADIUS)
TANGENT_RADIUS = math.hypot(RADIUS, LAMP_OFFSET)
LAMP_RADIUS = np.array([RADIUS - LAMP_OFFSET, RADIUS + LAMP_OFFSET, TANGENT_RADIUS, TANGENT_RADIUS])
LAMP_ANGLES = np.array([0., 0., TANGENT_ANGLE, -TANGENT_ANGLE])
MATCH_DISTANCE = 0.014


@dataclass(frozen=True)
class Detection:
    angle: float
    state: str
    world_center: np.ndarray
    pixel_center: np.ndarray
    lamp_pixels: np.ndarray
    rotation_vector: np.ndarray | None = None
    translation: np.ndarray | None = None
    reprojection_error: float | None = None


def _color_components(frame, camera, scene):
    # Bound processing to the projected wheel, which also excludes field markings.
    outline = np.array([[x, -LAMP_FRONT * scene.sign, AXIS[1] + z] for x in (-0.44, 0.44) for z in (-0.44, 0.44)])
    if np.min(camera.to_optical(outline)[:, 2]) < 0.1:
        return np.empty((0, 2))
    projection = camera.project(outline)
    start = np.maximum(0, np.floor(projection.min(axis=0)).astype(int) - 5)
    end = np.minimum((frame.shape[1], frame.shape[0]), np.ceil(projection.max(axis=0)).astype(int) + 5)
    if np.any(end <= start):
        return np.empty((0, 2))
    x, y = start
    right, bottom = end
    hsv = cv2.cvtColor(frame[y:bottom, x:right], cv2.COLOR_BGR2HSV)
    if scene.side == 'red':
        mask = cv2.inRange(hsv, (0, 95, 115), (17, 255, 255))
        mask |= cv2.inRange(hsv, (166, 95, 115), (179, 255, 255))
    else:
        mask = cv2.inRange(hsv, (88, 95, 115), (143, 255, 255))
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    area = stats[1:, cv2.CC_STAT_AREA]
    size_scale = (camera.intrinsics.matrix[0, 0] / 1108.) ** 2
    valid = (area >= max(3, 15 * size_scale)) & (area <= 940 * size_scale)
    return centroids[1:][valid] + start


def _pose(camera, scene, angle, pixels):
    rotation = camera.optical_to_world.T @ plate_rotation(angle)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = camera.to_optical(scene.target(angle, depth=LAMP_FRONT)).reshape(3, 1)
    ok, rvec, tvec = cv2.solvePnP(
        LAMP_POINTS, pixels, camera.intrinsics.matrix, camera.intrinsics.distortion,
        rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        return None, None, None
    projected, _ = cv2.projectPoints(LAMP_POINTS, rvec, tvec, camera.intrinsics.matrix, camera.intrinsics.distortion)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
    return rvec.ravel(), tvec.ravel(), rms


def detect(frame, camera, scene):
    pixels = _color_components(frame, camera, scene)
    if len(pixels) < 2:
        return []
    relative = camera.intersect_lamp_plane(pixels, scene)[:, (0, 2)] - AXIS
    radii = np.linalg.norm(relative, axis=1)
    good = (radii > 0.277) & (radii < 0.423)
    relative, radii, pixels = relative[good], radii[good], pixels[good]
    if len(pixels) < 2:
        return []
    theta = np.arctan2(relative[:, 1], relative[:, 0])
    seeds = theta[np.abs(radii - TANGENT_RADIUS) < MATCH_DISTANCE]
    candidates = []
    for angle in (seeds[:, None] - LAMP_ANGLES[2:]).ravel():
        positions = LAMP_RADIUS[:, None] * np.column_stack((np.cos(angle + LAMP_ANGLES), np.sin(angle + LAMP_ANGLES)))
        distances = np.linalg.norm(positions[:, None] - relative[None, :], axis=2)
        indices = distances.argmin(axis=1)
        residuals = distances[np.arange(4), indices]
        matched = residuals < MATCH_DISTANCE
        count = int(matched.sum())
        if not matched[2:].all() or count not in (2, 4):
            continue
        lamps = indices[matched]
        if len(set(lamps)) != count:
            continue
        angles = theta[lamps] - LAMP_ANGLES[matched]
        refined = math.atan2(np.sin(angles).sum(), np.cos(angles).sum())
        candidates.append((-count, float(residuals[matched].mean()), refined, lamps))

    detections = []
    used = set()
    for neg_count, _, angle, lamps in sorted(candidates, key=lambda item: item[:2]):
        if used.intersection(lamps):
            continue
        if detections and abs(math.remainder(angle - detections[0].angle, TARGET_STEP)) > 0.065:
            continue
        if neg_count == -4 and any(item.state == 'pending' for item in detections):
            continue
        lamp_pixels = pixels[lamps]
        state = 'pending' if neg_count == -4 else 'active'
        center = scene.target(angle)
        pixel_center = camera.project([center])[0]
        pose = _pose(camera, scene, angle, lamp_pixels) if state == 'pending' else (None, None, None)
        detections.append(Detection(angle, state, center, pixel_center, lamp_pixels, *pose))
        used.update(lamps)
    return detections
