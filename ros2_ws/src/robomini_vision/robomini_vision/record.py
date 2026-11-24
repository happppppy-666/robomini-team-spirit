"""Independent Gazebo evidence recorder. It publishes no control messages."""

import json
from pathlib import Path
import sys

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from robomini_interfaces.msg import RuneHit, RuneState, ScoreBoard
from sensor_msgs.msg import Image

from .node import image_bgr, seconds
from .overlay import label


MODES = ('fixed', 'random_fixed', 'small', 'big')


class Recorder(Node):
    def __init__(self):
        super().__init__('robomini_midterm_recorder')
        default_path = str(Path.cwd() / 'midterm-evidence')
        self.out = Path(self.declare_parameter('output_dir', default_path).value).expanduser()
        self.out.mkdir(parents=True, exist_ok=True)
        self.show = bool(self.declare_parameter('display', False).value)
        self.mode = 'waiting'
        self.hits = [0] * 4
        self.shots = 0
        self.score = 0
        self.first_complete = None
        self.finished = False
        self.events = []
        self.video = None
        self.frames = 0
        self.fps = 30.
        self.latest = None
        self.next_video_frame = None
        self.create_subscription(RuneState, '/rune/state', self.on_state, 10)
        self.create_subscription(RuneHit, '/rune/hit', self.on_hit, 20)
        self.create_subscription(ScoreBoard, '/score', self.on_score, 10)
        self.create_subscription(Image, '/vision/annotated', self.on_image, qos_profile_sensor_data)

    def on_state(self, msg):
        self.mode = msg.mode

    def on_hit(self, msg):
        item = {'sim_time_s': seconds(msg.stamp), 'target': int(msg.target),
                'valid': msg.valid, 'points': msg.score, 'ball': msg.ball}
        self.events.append(item)
        self.get_logger().info(json.dumps(item))

    def on_score(self, msg):
        new_hits = list(msg.mode_hits)
        if self.hits != new_hits:
            self.get_logger().info(f'Official Gazebo hits {dict(zip(MODES, new_hits))}')
        self.hits, self.shots, self.score = new_hits, msg.balls_fired, msg.total
        now = self.get_clock().now().nanoseconds * 1e-9
        if min(self.hits) < 5:
            self.first_complete = None
        elif self.first_complete is None:
            self.first_complete = now
        elif now - self.first_complete > 0.8:
            self.finished = True

    def on_image(self, msg):
        t = seconds(msg.header.stamp)
        if self.next_video_frame is not None and t + 1e-5 < self.next_video_frame:
            return
        image = image_bgr(msg)
        h, w = image.shape[:2]
        if self.video is None:
            self.video = cv2.VideoWriter(
                str(self.out / 'gazebo-midterm.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h),
            )
            if not self.video.isOpened():
                raise RuntimeError(f'Cannot create MP4 in {self.out}')
        label(image, 'OFFICIAL GAZEBO SCOREBOARD', 20, h - 61, (255, 230, 140), 0.56)
        label(image, f'{self.mode.upper()}   |   {self.score} pts   |   {self.shots} shots   |   Sim {t:.2f} s', 20, h - 35)
        label(image, '   '.join(f'{mode}: {hit}/5' for mode, hit in zip(MODES, self.hits)), 20, h - 12, size=0.57)
        if self.next_video_frame is None:
            self.next_video_frame = t
        skipped = max(0, min(5, int((t - self.next_video_frame) * self.fps)))
        for _ in range(skipped):
            self.video.write(self.latest)
            self.frames += 1
        self.video.write(image)
        self.latest = image
        self.frames += 1
        self.next_video_frame = max(t - 0.0001, self.next_video_frame + skipped / self.fps) + 1. / self.fps
        if self.show:
            cv2.imshow('RoboMini midterm | Gazebo', image)
            cv2.waitKey(1)

    def close(self):
        if self.video is not None:
            self.video.release()
        if self.show:
            cv2.destroyAllWindows()
        result = {'source': 'Official Gazebo referee ROS topics',
                  'complete': min(self.hits) >= 5 and self.frames > 0,
                  'mode_hits': dict(zip(MODES, self.hits)), 'balls_fired': self.shots,
                  'score': self.score, 'recorded_frames': self.frames, 'hit_events': self.events}
        (self.out / 'official-results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        return result['complete']


def main(args=None):
    cv2.setNumThreads(2)
    rclpy.init(args=args)
    node = Recorder()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        passed = node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not passed:
        sys.exit(1)
