"""ROS 2 adapter for the official Gazebo camera and gimbal interface."""

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from robomini_interfaces.msg import GimbalToVision, VisionToGimbal
from sensor_msgs.msg import CameraInfo, Image

from .detector import detect
from .geometry import Camera, Intrinsics, Scene
from .overlay import annotate
from .tracking import WheelTracker, plan


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def image_bgr(msg):
    if msg.encoding not in ('bgr8', 'rgb8') or msg.step < 3 * msg.width:
        raise ValueError(f'Unsupported ROS image: {msg.encoding}, stride {msg.step}')
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    frame = rows[:, :3 * msg.width].reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else frame.copy()


@dataclass(frozen=True)
class Feedback:
    time: float
    yaw: float
    pitch: float
    yaw_velocity: float
    pitch_velocity: float
    speed: float


class VisionNode(Node):
    def __init__(self):
        super().__init__('robomini_vision')
        self.scene = Scene(self.declare_parameter('side', 'blue').value)
        self.fire = self.declare_parameter('enable_fire', False).value
        self.publish_preview = self.declare_parameter('publish_preview', True).value
        self.command_delay = self.declare_parameter('command_delay_s', 0.012).value
        self.max_miss = self.declare_parameter('max_miss_m', 0.043).value
        self.intrinsics = None
        self.calibration_size = None
        self.feedback = deque(maxlen=80)
        self.tracker = WheelTracker()
        self.last_frame = -math.inf
        self.last_fire = -math.inf
        self.previous_pending = None
        self.stable_since = None
        self.last_seen_pending = -math.inf
        self.command_pub = self.create_publisher(VisionToGimbal, '/vision_to_gimbal', 5)
        self.preview_pub = self.create_publisher(Image, '/vision/annotated', qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_calibration, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(GimbalToVision, '/gimbal_to_vision', self.on_feedback, 5)
        self.get_logger().info(f"side={self.scene.side}, enable_fire={self.fire}")

    def time(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_calibration(self, msg):
        matrix = np.array(msg.k, dtype=float).reshape(3, 3)
        if np.isfinite(matrix).all() and matrix[0, 0] > 0 and matrix[1, 1] > 0:
            self.intrinsics = Intrinsics(matrix, np.asarray(msg.d, dtype=float))
            self.calibration_size = (msg.height, msg.width)

    def on_feedback(self, msg):
        values = (msg.yaw, msg.pitch, msg.yaw_vel, msg.pitch_vel, msg.bullet_speed)
        if not all(math.isfinite(v) for v in values) or msg.bullet_speed <= 0:
            return
        now = self.time()
        if self.feedback and now < self.feedback[-1].time:
            self.feedback.clear()
        self.feedback.append(Feedback(now, *values))

    def pose_at(self, stamp):
        if not self.feedback:
            return None
        closest = min(self.feedback, key=lambda state: abs(state.time - stamp))
        if abs(closest.time - stamp) > 0.09:
            return None
        dt = max(-0.020, min(0.020, stamp - closest.time))
        return closest.yaw + dt * closest.yaw_velocity, closest.pitch + dt * closest.pitch_velocity

    def command(self, pending, motion, stamp, now):
        current = self.feedback[-1]
        previous = self.previous_pending
        if previous is None or abs(math.remainder(pending.angle - previous, 2 * math.pi)) > 0.28:
            self.stable_since = stamp
        self.previous_pending = pending.angle
        if self.stable_since is None:
            self.stable_since = stamp
        ahead = min(0.035, max(0., now - current.time) + self.command_delay)
        aim = plan(self.scene, pending.angle, motion, stamp, now,
                   current.yaw + current.yaw_velocity * ahead,
                   current.pitch + current.pitch_velocity * ahead,
                   current.speed, self.command_delay)
        ready = (self.fire and pending.reprojection_error is not None and pending.reprojection_error < 2.5
                 and aim.miss < self.max_miss and stamp - self.stable_since >= 0.042
                 and now - self.last_fire >= 0.37 and now - stamp < 0.10)
        self.command_pub.publish(VisionToGimbal(
            mode=2 if ready else 1, yaw=aim.yaw, pitch=aim.pitch,
            yaw_vel=aim.yaw_velocity, pitch_vel=aim.pitch_velocity,
        ))
        if ready:
            self.last_fire = now
        return aim, 'FIRE REQUEST' if ready else ('TRACKING' if self.fire else 'AIM ONLY')

    def on_image(self, msg):
        try:
            image = image_bgr(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return
        stamp, now = seconds(msg.header.stamp), self.time()
        if stamp <= 0. or now <= 0. or stamp < now - 0.20 or stamp > now + 0.08:
            return
        if stamp < self.last_frame:
            self.tracker = WheelTracker()
            self.last_fire = -math.inf
            self.previous_pending = None
            self.stable_since = None
            self.last_seen_pending = -math.inf
        if stamp == self.last_frame:
            return
        self.last_frame = stamp
        pose = self.pose_at(stamp)
        available = pose is not None and self.intrinsics is not None and image.shape[:2] == self.calibration_size
        detections = detect(image, Camera.on_gimbal(self.scene, self.intrinsics, *pose), self.scene) if available else []
        motion = self.tracker.observe(stamp, detections)
        pending = next((item for item in detections if item.state == 'pending'), None)
        aim = None
        status = 'WAITING FOR CALIBRATION / GIMBAL' if not available else 'SEARCHING'
        if pending is None:
            self.previous_pending = None
            self.stable_since = None
            if available and stamp - self.last_seen_pending > 0.18:
                self.command_pub.publish(VisionToGimbal(mode=1, yaw=0., pitch=0.20))
                status = 'CENTERING ON WHEEL'
        else:
            self.last_seen_pending = stamp
            if motion is not None and now - self.feedback[-1].time < 0.10:
                aim, status = self.command(pending, motion, stamp, now)
        if self.publish_preview:
            annotate(image, detections, self.scene, pose, aim, 'GAZEBO CAMERA', status)
            self.preview_pub.publish(Image(
                header=msg.header, height=msg.height, width=msg.width,
                encoding='bgr8', step=msg.width * 3, data=image.tobytes(),
            ))


def main(args=None):
    cv2.setNumThreads(2)
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
