"""Estimate wheel motion from lamp observations, including target handoffs."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .geometry import TARGET_STEP, flight_solution, impact


@dataclass(frozen=True)
class Motion:
    origin: float
    velocity: float
    acceleration: float = 0.0
    omega: float = 0.0
    sine: float = 0.0
    cosine: float = 0.0

    def offset(self, start, end):
        a, b = start - self.origin, end - self.origin
        if not self.omega:
            return self.velocity * (b - a) + 0.5 * self.acceleration * (b * b - a * a)
        return (self.velocity * (b - a)
                + self.sine * (math.sin(self.omega * b) - math.sin(self.omega * a))
                + self.cosine * (math.cos(self.omega * b) - math.cos(self.omega * a)))


class WheelTracker:
    def __init__(self):
        self.history = deque(maxlen=280)
        self.periodic = None
        self.last_fit = -math.inf
        self.active_count = 0

    def observe(self, stamp, detections):
        if not detections:
            return None
        previous_time, previous_phase = self.history[-1] if self.history else (stamp, detections[0].angle)
        delta = np.median([math.remainder(d.angle - previous_phase, TARGET_STEP) for d in detections])
        active = sum(d.state == 'active' for d in detections)
        dt = stamp - previous_time
        if dt < 0 or dt > 0.35 or abs(delta) > 0.055 + 2.5 * dt or active + 2 <= self.active_count:
            self.history.clear()
            self.periodic = None
            self.last_fit = -math.inf
        self.active_count = active
        if not self.history or stamp > previous_time:
            self.history.append((stamp, previous_phase + float(delta)))
        while self.history[0][0] < stamp - 2.65:
            self.history.popleft()
        samples = np.asarray(self.history)
        time = samples[:, 0] - stamp
        phase = samples[:, 1] - samples[-1, 1]
        if len(time) < 4 or -time[0] < 0.075:
            return None
        if stamp - self.last_fit >= 0.11:
            self.periodic = self._fit_periodic(stamp, time, phase) if -time[0] > 1.08 else None
            self.last_fit = stamp
        if self.periodic is not None:
            return self.periodic
        recent = time > -0.34
        time, phase = time[recent], phase[recent]
        if len(time) < 4 or -time[0] < 0.075:
            return None
        slope, _ = np.polyfit(time, phase, 1)
        if abs(slope) < 0.06 and np.ptp(phase) < 0.018:
            return Motion(stamp, 0.)
        if -time[0] < 0.17:
            return Motion(stamp, float(np.clip(slope, -2.20, 2.20)))
        curvature, velocity, _ = np.polyfit(time, phase, 2)
        acceleration = float(np.clip(2.0 * curvature, -2.35, 2.35))
        if abs(abs(slope) - math.pi / 3) < 0.07 and abs(acceleration) < 0.65:
            return Motion(stamp, math.copysign(math.pi / 3, slope))
        return Motion(stamp, float(np.clip(velocity, -2.20, 2.20)), acceleration)

    @staticmethod
    def _fit_periodic(stamp, time, phase):
        best = None
        for omega in np.linspace(1.884, 2.0, 7):
            basis = np.column_stack((np.ones(len(time)), time, np.sin(omega * time), np.cos(omega * time)))
            coeff = np.linalg.lstsq(basis, phase, rcond=None)[0]
            _, mean, sine, cosine = coeff
            amplitude = omega * math.hypot(sine, cosine)
            if not 0.65 <= amplitude <= 1.16 or abs(abs(mean) + amplitude - 2.09) > 0.17:
                continue
            residual = float(np.sqrt(np.mean((basis @ coeff - phase) ** 2)))
            if residual < 0.018 and (best is None or residual < best[0]):
                best = residual, Motion(stamp, float(mean), omega=float(omega), sine=float(sine), cosine=float(cosine))
        return best[1] if best else None


@dataclass(frozen=True)
class Aim:
    yaw: float
    pitch: float
    yaw_velocity: float
    pitch_velocity: float
    flight_time: float
    miss: float


def plan(scene, angle, motion, frame_time, now, yaw, pitch, speed, command_delay=0.012):
    launch_time = now + command_delay

    def solve(at):
        flight = 0.16
        for _ in range(4):
            point = scene.target(angle + motion.offset(frame_time, at + flight))
            target_yaw, target_pitch, flight = flight_solution(scene, point, speed)
        return target_yaw, target_pitch, flight

    desired_yaw, desired_pitch, flight = solve(launch_time)
    next_yaw, next_pitch, _ = solve(launch_time + 0.025)
    current_impact = impact(scene, yaw, pitch, speed)
    miss = math.inf
    if current_impact is not None:
        point, current_flight = current_impact
        predicted = scene.target(angle + motion.offset(frame_time, launch_time + current_flight))
        miss = float(np.linalg.norm(point - predicted))
    return Aim(desired_yaw, desired_pitch, (next_yaw - desired_yaw) / 0.025,
               (next_pitch - desired_pitch) / 0.025, flight, miss)
