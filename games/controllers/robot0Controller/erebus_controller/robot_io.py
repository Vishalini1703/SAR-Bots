"""Webots device I/O boundary for the Erebus controller."""

from __future__ import annotations

import math
import struct
from typing import Any

import numpy as np

from . import config
from .utils.frames import normalize_angle, yaw_from_compass


class RobotIO:
    _EPUCK_IR_CLEAR_RAW_MAX = 72.0
    _EPUCK_IR_TABLE = (
        (0.0, 4095.0),
        (0.005, 2133.33),
        (0.01, 1465.73),
        (0.015, 601.46),
        (0.02, 383.84),
        (0.03, 234.93),
        (0.04, 158.03),
        (0.05, 120.0),
        (0.06, 104.09),
        (0.07, 67.19),
    )
    _EPUCK_TOF_TABLE = (
        (0.00, 19.8),
        (0.05, 58.5),
        (0.10, 111.0),
        (0.20, 218.9),
        (0.50, 531.9),
        (1.00, 1052.0),
        (1.70, 1780.5),
        (2.00, 2000.0),
    )

    def __init__(self, robot: Any, cfg=config) -> None:
        self.robot = robot
        self.cfg = cfg
        self.time_step_ms = self._resolve_time_step_ms()
        self._available_device_names = self._list_device_names()

        self.left_motor = self._get_required_device(
            ["left wheel motor", "wheel1 motor", "left_motor", "motor_left"]
        )
        self.right_motor = self._get_required_device(
            ["right wheel motor", "wheel2 motor", "right_motor", "motor_right"]
        )
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_wheel_sensor = self._get_optional_device(["left wheel sensor", "wheel1 sensor"])
        if self.left_wheel_sensor is not None and hasattr(self.left_wheel_sensor, "enable"):
            self.left_wheel_sensor.enable(self.time_step_ms)

        self.right_wheel_sensor = self._get_optional_device(["right wheel sensor", "wheel2 sensor"])
        if self.right_wheel_sensor is not None and hasattr(self.right_wheel_sensor, "enable"):
            self.right_wheel_sensor.enable(self.time_step_ms)

        self.gps = self._get_optional_device(["gps"])
        if self.gps is not None:
            self.gps.enable(self.time_step_ms)

        self.imu = self._get_optional_device(["inertial unit", "imu"])
        if self.imu is not None:
            self.imu.enable(self.time_step_ms)

        self.gyro = self._get_optional_device(["gyro", "Gyro"])
        if self.gyro is not None:
            self.gyro.enable(self.time_step_ms)

        self.compass = self._get_optional_device(["compass"])
        if self.compass is not None:
            self.compass.enable(self.time_step_ms)

        self.camera = self._get_optional_device(["camera_centre", "camera_center", "camera", "Camera"])
        if self.camera is not None and hasattr(self.camera, "enable"):
            self.camera.enable(self.time_step_ms)
        self.color_sensor = self._get_optional_device(
            ["colour_sensor", "color_sensor", "color", "colour", "color sensor", "colour sensor"]
        )
        if self.color_sensor is not None and hasattr(self.color_sensor, "enable"):
            self.color_sensor.enable(self.time_step_ms)

        self.emitter = self._get_required_device(["emitter"])
        self.receiver = self._get_optional_device(["receiver"])
        if self.receiver is not None:
            self.receiver.enable(self.time_step_ms)

        self.range_sensors = self._init_range_sensors()
        self._odometry_yaw: float | None = None
        self._last_wheel_pose_for_odometry: tuple[float, float] | None = None
        self._last_gyro_yaw_delta: float | None = None
        self._last_gps_pose_for_heading: tuple[float, float] | None = None
        self._last_requested_wheel_speeds = (0.0, 0.0)

    def _resolve_time_step_ms(self) -> int:
        try:
            basic = int(self.robot.getBasicTimeStep())
        except Exception:
            basic = 0
        if basic > 0:
            return basic
        return int(self.cfg.TIME_STEP_MS)

    def _get_device_if_present(self, name: str) -> Any | None:
        if name not in self._available_device_names:
            return None
        try:
            return self.robot.getDevice(name)
        except Exception:
            return None

    def _list_device_names(self) -> set[str]:
        names: set[str] = set()
        try:
            count = int(self.robot.getNumberOfDevices())
        except Exception:
            return names
        for idx in range(count):
            try:
                dev = self.robot.getDeviceByIndex(idx)
            except Exception:
                continue
            if dev is None:
                continue
            try:
                name = dev.getName()
            except Exception:
                continue
            if isinstance(name, str) and name:
                names.add(name)
        return names

    def _get_optional_device(self, names: list[str]) -> Any | None:
        for name in names:
            dev = self._get_device_if_present(name)
            if dev is not None:
                return dev
        return None

    def _get_required_device(self, names: list[str]) -> Any:
        dev = self._get_optional_device(names)
        if dev is None:
            raise RuntimeError(f"Required device not found. Tried names: {names}")
        return dev

    def _init_range_sensors(self) -> dict[str, tuple[Any, float]]:
        # Relative angles are robot-local yaw offsets (left positive).
        # The default world now uses a Webots E-puck base, so prefer its
        # forward-facing sensor ring and ignore the rear pair for control.
        candidates = {
            "tof": 0.0,
            "ps0": -0.30,
            "ps1": -0.80,
            "ps2": -1.57,
            "ps5": 1.57,
            "ps6": 0.80,
            "ps7": 0.30,
            "leftDist": math.pi / 2.0,
            "frontDist": 0.0,
            "rightDist": -math.pi / 2.0,
            "left ds": math.pi / 2.0,
            "front ds": 0.0,
            "right ds": -math.pi / 2.0,
            "left distance sensor": math.pi / 2.0,
            "front distance sensor": 0.0,
            "right distance sensor": -math.pi / 2.0,
        }
        sensors: dict[str, tuple[Any, float]] = {}
        for name, angle in candidates.items():
            dev = self._get_optional_device([name])
            if dev is None or not hasattr(dev, "enable") or not hasattr(dev, "getValue"):
                continue
            dev.enable(self.time_step_ms)
            sensors[name] = (dev, angle)
        if not sensors:
            raise RuntimeError("No distance sensors found")
        return sensors

    def step(self) -> bool:
        return self.robot.step(self.time_step_ms) != -1

    def set_wheel_speeds(self, left: float, right: float) -> None:
        max_speed = self.cfg.MAX_WHEEL_SPEED
        left = max(-max_speed, min(max_speed, left))
        right = max(-max_speed, min(max_speed, right))
        self._last_requested_wheel_speeds = (left, right)
        self.left_motor.setVelocity(left * self.cfg.LEFT_MOTOR_DIRECTION)
        self.right_motor.setVelocity(right * self.cfg.RIGHT_MOTOR_DIRECTION)

    def read_pose(self) -> tuple[float, float, float]:
        x_w = 0.0
        z_w = 0.0
        if self.gps is not None:
            gps_values = self.gps.getValues()
            x_w = float(gps_values[0])
            z_w = float(gps_values[2])

        compass_yaw = None
        if self.compass is not None:
            compass_yaw = yaw_from_compass(
                self.compass.getValues(),
                planar_axes=getattr(self.cfg, "COMPASS_PLANAR_AXES", "xz"),
            )
        if hasattr(self, "robot") and getattr(self.cfg, "USE_COMPASS_ONLY", False):
            yaw = 0.0 if compass_yaw is None else compass_yaw
            return x_w, z_w, yaw
        yaw = self._update_odometry_yaw(compass_yaw)
        if yaw is None:
            yaw = 0.0
        yaw = self._apply_gps_motion_yaw(x_w, z_w, yaw)
        return x_w, z_w, yaw

    def _apply_gps_motion_yaw(self, x_w: float, z_w: float, yaw: float) -> float:
        if not getattr(self.cfg, "USE_GPS_MOTION_YAW", False):
            return yaw
        previous = self._last_gps_pose_for_heading
        self._last_gps_pose_for_heading = (x_w, z_w)
        if previous is None:
            return yaw
        dx = x_w - previous[0]
        dz = z_w - previous[1]
        if math.hypot(dx, dz) < getattr(self.cfg, "GPS_YAW_MIN_DELTA_M", 0.004):
            return yaw
        left_cmd, right_cmd = self._last_requested_wheel_speeds
        if left_cmd * right_cmd < 0.0 and abs(left_cmd - right_cmd) > 0.1:
            return yaw
        motion_yaw = math.atan2(dz, dx)
        if left_cmd < -0.05 and right_cmd < -0.05:
            motion_yaw = normalize_angle(motion_yaw + math.pi)
        return motion_yaw

    def _update_odometry_yaw(self, compass_yaw: float | None) -> float | None:
        left, right = self.read_wheel_positions()
        if left is None or right is None:
            gyro_delta = self._read_gyro_yaw_delta()
            if gyro_delta is not None:
                if self._odometry_yaw is None:
                    self._odometry_yaw = 0.0 if compass_yaw is None else float(compass_yaw)
                self._odometry_yaw = normalize_angle(self._odometry_yaw + gyro_delta)
                return self._odometry_yaw
            return compass_yaw

        if self._odometry_yaw is None:
            self._odometry_yaw = 0.0 if compass_yaw is None else float(compass_yaw)
            self._last_wheel_pose_for_odometry = (left, right)
            return self._odometry_yaw

        if self._last_wheel_pose_for_odometry is None:
            self._last_wheel_pose_for_odometry = (left, right)
            return self._odometry_yaw

        prev_left, prev_right = self._last_wheel_pose_for_odometry
        self._last_wheel_pose_for_odometry = (left, right)

        delta_left = (left - prev_left) * self.cfg.WHEEL_RADIUS_M * self.cfg.LEFT_MOTOR_DIRECTION
        delta_right = (right - prev_right) * self.cfg.WHEEL_RADIUS_M * self.cfg.RIGHT_MOTOR_DIRECTION
        delta_yaw = (delta_right - delta_left) / max(self.cfg.WHEEL_TRACK_M, 1e-6)
        self._odometry_yaw = normalize_angle(self._odometry_yaw + delta_yaw)

        if compass_yaw is None:
            return self._odometry_yaw

        compass_err = normalize_angle(compass_yaw - self._odometry_yaw)
        if abs(compass_err) < 0.20:
            self._odometry_yaw = normalize_angle(self._odometry_yaw + 0.1 * compass_err)
        return self._odometry_yaw

    def _read_gyro_yaw_delta(self) -> float | None:
        gyro = getattr(self, "gyro", None)
        if gyro is None or not hasattr(gyro, "getValues"):
            return None
        try:
            values = gyro.getValues()
            yaw_rate = float(values[1])
        except Exception:
            return None
        return yaw_rate * (self.time_step_ms / 1000.0)

    @staticmethod
    def _interpolate_distance(raw: float, table: tuple[tuple[float, float], ...]) -> float:
        by_raw = sorted((raw_value, distance_m) for distance_m, raw_value in table)
        if raw <= by_raw[0][0]:
            return by_raw[0][1]
        if raw >= by_raw[-1][0]:
            return by_raw[-1][1]
        for idx in range(1, len(by_raw)):
            raw_hi, dist_hi = by_raw[idx]
            raw_lo, dist_lo = by_raw[idx - 1]
            if raw > raw_hi:
                continue
            span = raw_hi - raw_lo
            ratio = 0.0 if span == 0.0 else (raw - raw_lo) / span
            return dist_lo + ratio * (dist_hi - dist_lo)
        return by_raw[-1][1]

    def _distance_to_meters(self, name: str, raw: float) -> float:
        if name == "tof":
            return min(self.cfg.SCAN_MAX_RANGE_M, self._interpolate_distance(raw, self._EPUCK_TOF_TABLE))
        if name.startswith("ps"):
            if 0.0 <= raw <= 2.0:
                return min(self.cfg.SCAN_MAX_RANGE_M, max(0.005, raw))
            if raw <= self._EPUCK_IR_CLEAR_RAW_MAX:
                return self.cfg.SCAN_MAX_RANGE_M
            return min(self.cfg.SCAN_MAX_RANGE_M, self._interpolate_distance(raw, self._EPUCK_IR_TABLE))
        if raw <= 0.0:
            return self.cfg.SCAN_MAX_RANGE_M
        if raw < 5.0:
            return min(self.cfg.SCAN_MAX_RANGE_M, raw)
        normalized = max(0.0, min(1.0, raw / 4096.0))
        return max(0.01, self.cfg.SCAN_MAX_RANGE_M * (1.0 - normalized))

    def read_scan_rays(self) -> list[tuple[float, float]]:
        rays: list[tuple[float, float]] = []
        for name, (sensor, rel_angle) in self.range_sensors.items():
            distance = self._distance_to_meters(name, float(sensor.getValue()))
            rays.append((rel_angle, distance))
        return rays

    def read_ranges(self) -> dict[str, float]:
        rays = self.read_scan_rays()
        front = self.cfg.SCAN_MAX_RANGE_M
        left = self.cfg.SCAN_MAX_RANGE_M
        right = self.cfg.SCAN_MAX_RANGE_M
        for angle, dist in rays:
            if abs(angle) < 0.40:
                front = min(front, dist)
            elif angle > 0.0:
                left = min(left, dist)
            else:
                right = min(right, dist)
        return {"left": left, "front": front, "right": right}

    def read_wheel_positions(self) -> tuple[float | None, float | None]:
        left = None
        right = None
        left_sensor = getattr(self, "left_wheel_sensor", None)
        right_sensor = getattr(self, "right_wheel_sensor", None)
        if left_sensor is not None and hasattr(left_sensor, "getValue"):
            left = float(left_sensor.getValue())
        if right_sensor is not None and hasattr(right_sensor, "getValue"):
            right = float(right_sensor.getValue())
        return left, right

    @staticmethod
    def _read_camera_image(camera: Any | None) -> np.ndarray | None:
        if camera is None:
            return None
        width = int(camera.getWidth())
        height = int(camera.getHeight())
        raw = camera.getImage()
        if raw is None:
            return None
        img = np.frombuffer(raw, dtype=np.uint8)
        if img.size != width * height * 4:
            return None
        bgra = img.reshape((height, width, 4))
        rgb = bgra[:, :, :3][:, :, ::-1].copy()
        return rgb

    def read_camera_rgb(self) -> np.ndarray | None:
        return self._read_camera_image(self.camera)

    def read_floor_rgb(self) -> np.ndarray | None:
        return self._read_camera_image(self.color_sensor)

    def send_victim_report(self, x_w: float, z_w: float, victim_type: str) -> None:
        x_cm = int(round(x_w * 100.0))
        z_cm = int(round(z_w * 100.0))
        victim_byte = victim_type[:1].encode("ascii", errors="ignore")
        if len(victim_byte) != 1:
            raise ValueError("victim_type must contain an ASCII character")
        payload = struct.pack("i i c", x_cm, z_cm, victim_byte)
        self.emitter.send(payload)

    def request_game_info(self) -> None:
        self.emitter.send(struct.pack("c", b"G"))

    def request_lack_of_progress(self) -> None:
        self.emitter.send(struct.pack("c", b"L"))

    def request_exit(self) -> None:
        self.emitter.send(struct.pack("c", b"E"))

    def send_map_submission(self, rows: list[list[str]]) -> None:
        if not rows or not rows[0]:
            raise ValueError("map rows must be a non-empty 2D list")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("all map rows must have the same width")
        shape = struct.pack("2i", len(rows), width)
        flat_map = ",".join(cell for row in rows for cell in row).encode("utf-8")
        self.emitter.send(shape + flat_map)
        self.emitter.send(struct.pack("c", b"M"))

    def flush_receiver(self) -> list[bytes]:
        packets: list[bytes] = []
        if self.receiver is None:
            return packets
        while self.receiver.getQueueLength() > 0:
            packets.append(bytes(self.receiver.getBytes()))
            self.receiver.nextPacket()
        return packets

    def read_supervisor_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for packet in self.flush_receiver():
            if len(packet) == 1:
                try:
                    code = packet.decode("utf-8")
                except UnicodeDecodeError:
                    code = ""
                messages.append({"kind": "signal", "code": code, "raw": packet})
                continue
            if len(packet) == struct.calcsize("c f i i"):
                try:
                    code_b, score, sim_time, real_time = struct.unpack("c f i i", packet)
                    code = code_b.decode("utf-8")
                except (struct.error, UnicodeDecodeError):
                    messages.append({"kind": "raw", "raw": packet})
                    continue
                if code == "G":
                    messages.append(
                        {
                            "kind": "game_info",
                            "score": float(score),
                            "remaining_sim_time": int(sim_time),
                            "remaining_real_time": int(real_time),
                            "raw": packet,
                        }
                    )
                    continue
            messages.append({"kind": "raw", "raw": packet})
        return messages
