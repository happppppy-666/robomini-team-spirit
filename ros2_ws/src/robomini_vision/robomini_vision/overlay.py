import math

import cv2
import numpy as np


def label(image, message, x, y, color=(235, 240, 246), size=0.66):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (21, 24, 29), 4, cv2.LINE_AA)
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def annotate(image, detections, scene, pose, aim, source, status):
    h, w = image.shape[:2]
    cv2.drawMarker(image, (w // 2, h // 2), (226, 228, 230), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
    for item in detections:
        color = (43, 206, 255) if item.state == 'pending' else (110, 235, 138)
        x, y = (int(round(v)) for v in item.pixel_center)
        for u, v in np.rint(item.lamp_pixels).astype(int):
            cv2.circle(image, (u, v), 4, color, 1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 32, color, 2, cv2.LINE_AA)
        label(image, item.state.upper(), x - 34, y - 40, color, 0.51)
    label(image, f'{source}  |  {scene.side.upper()}  |  {status}', 20, 30)
    pending = next((item for item in detections if item.state == 'pending'), None)
    if pending is None or pose is None:
        return image
    delta_yaw, delta_pitch = map(math.degrees, scene.angles_to(pending.world_center, *pose))
    xyz = ', '.join(f'{v:.3f}' for v in pending.world_center)
    label(image, f'World XYZ ({xyz}) m   Delta yaw {delta_yaw:+.2f} deg   Delta pitch {delta_pitch:+.2f} deg', 20, 57, size=0.60)
    if pending.translation is not None:
        xyz = ', '.join(f'{v:.3f}' for v in pending.translation)
        r = ', '.join(f'{v:.2f}' for v in pending.rotation_vector)
        label(image, f'PnP optical XYZ ({xyz}) m   Rodrigues ({r}) rad   RMS {pending.reprojection_error:.2f} px', 20, 82, size=0.56)
    if aim is not None:
        label(image, f'Sim lead {math.degrees(aim.yaw):+.2f}/{math.degrees(aim.pitch):+.2f} deg   Flight {aim.flight_time * 1e3:.0f} ms   Error {aim.miss * 100.:.1f} cm', 20, 107, size=0.56)
    return image
