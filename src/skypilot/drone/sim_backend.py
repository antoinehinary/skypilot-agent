"""DroneBackend implementation on gym-pybullet-drones (Crazyflie 2.x, CF2X model).

Control loop: every command moves a position setpoint along a straight line at the
requested speed and feeds it, with its velocity as feed-forward, to the
cascaded DSL PID controller. Each `env.step()` advances physics_hz / control_hz
PyBullet steps. Commands block until the drone is within tolerance of the target
or the settle timeout expires.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import time
from collections.abc import Callable

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from skypilot.config import SimConfig, load_sim_config
from skypilot.drone.backend import DroneBackend, DroneState, MoveResult, Vec3

logger = logging.getLogger(__name__)

_DRONE_MODEL = DroneModel.CF2X
# Time given to the drone to drop onto the pad at start-up, and to come to rest after
# the motors are cut on landing.
_GROUND_SETTLE_S = 1.0

# Indices into the 20-element gym-pybullet-drones state vector.
_POS = slice(0, 3)
_YAW = 9
_VEL = slice(10, 13)


def _fmt(v: np.ndarray | Vec3) -> str:
    # round() + 0.0 turns -0.004 into 0.0 so messages never show "-0.00".
    return "({:.2f}, {:.2f}, {:.2f})".format(*(round(float(c), 2) + 0.0 for c in v))


class SimBackend(DroneBackend):
    """Single Crazyflie in PyBullet, starting at rest on the pad at the world origin.

    Args:
        config: Simulator parameters; defaults to `config/sim.yaml`.
        gui: Open the PyBullet GUI (demos) instead of running headless (tests, eval).
    """

    def __init__(self, config: SimConfig | None = None, gui: bool = False) -> None:
        self.config = config or load_sim_config()
        self._gui = gui
        # NOTE: gym-pybullet-drones prints to stdout during construction. Swallow it:
        # stdout must stay clean for the MCP stdio transport.
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self._env = CtrlAviary(
                drone_model=_DRONE_MODEL,
                num_drones=1,
                physics=Physics.PYB,
                pyb_freq=self.config.physics_hz,
                ctrl_freq=self.config.control_hz,
                gui=gui,
                obstacles=False,
                user_debug_gui=False,
            )
        logger.debug("CtrlAviary init output:\n%s", captured.getvalue())
        self._client = self._env.getPyBulletClient()
        self._ctrl = DSLPIDControl(drone_model=_DRONE_MODEL)
        self._dt = self._env.CTRL_TIMESTEP
        self._step_callbacks: list[Callable[[], None]] = []
        self._closed = False
        if gui:
            g = self.config.gui
            p.resetDebugVisualizerCamera(
                cameraDistance=g.camera_distance_m,
                cameraYaw=g.camera_yaw_deg,
                cameraPitch=g.camera_pitch_deg,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self._client,
            )

        self._obs, *_ = self._env.reset()
        self._obs = self._obs[0]
        self._flying = False
        self._battery_pct = self.config.battery.initial_pct
        # The default spawn height is slightly above the pad: let the drone drop and
        # record its resting height, then start the clock from zero.
        self._t0 = 0.0
        self._wall_anchor = (time.monotonic(), 0.0)
        self._run_steps(math.ceil(_GROUND_SETTLE_S / self._dt))
        self._rest_z = float(self._obs[2])
        self._t0 = self._raw_sim_time()
        self._setpoint = self._obs[_POS].copy()

    # ------------------------------------------------------------------ commands

    def takeoff(self, altitude_m: float) -> MoveResult:
        if self._flying:
            return self._fail("already flying")
        self._ctrl.reset()
        self._flying = True
        start = self._obs[_POS].copy()
        target = np.array([start[0], start[1], altitude_m])
        return self._move(start, target, self.config.motion.takeoff_speed_mps, "took off to")

    def goto(
        self, x_m: float, y_m: float, z_m: float, speed_mps: float | None = None
    ) -> MoveResult:
        if not self._flying:
            return self._fail("not flying: take off first")
        if speed_mps is None:
            speed_mps = self.config.motion.default_speed_mps
        return self._move(self._setpoint, np.array([x_m, y_m, z_m]), speed_mps, "reached")

    def hover(self, seconds: float) -> MoveResult:
        if not self._flying:
            return self._fail("not flying: take off first")
        t_start = self._start_command()
        self._run_steps(math.ceil(seconds / self._dt), self._setpoint)
        error = self._distance_to(self._setpoint)
        success = error <= self.config.motion.position_tolerance_m
        message = f"hovered at {_fmt(self._obs[_POS])}"
        if not success:
            message += f", drifted {error:.2f} m from {_fmt(self._setpoint)}"
        return self._result(success, t_start, message)

    def land(self) -> MoveResult:
        if not self._flying:
            return self._fail("already landed")
        motion = self.config.motion
        t_start = self._start_command()
        cutoff_z = self._rest_z + motion.motor_cutoff_height_m
        target = np.array([self._setpoint[0], self._setpoint[1], self._rest_z])
        self._ramp(
            self._setpoint, target, motion.landing_speed_mps, stop=lambda: self._obs[2] <= cutoff_z
        )
        if self._obs[2] > cutoff_z:
            self._settle(target, motion.settle_timeout_s, done=lambda: self._obs[2] <= cutoff_z)
        touched_down = self._obs[2] <= cutoff_z
        # The PID never commands zero thrust, so stop the motors explicitly.
        self._flying = False
        self._run_steps(math.ceil(_GROUND_SETTLE_S / self._dt))
        self._setpoint = self._obs[_POS].copy()
        if touched_down:
            return self._result(True, t_start, f"landed at {_fmt(self._obs[_POS])}")
        return self._result(False, t_start, "landing timed out, motors stopped in the air")

    def get_state(self) -> DroneState:
        return DroneState(
            position_m=tuple(float(v) for v in self._obs[_POS]),
            velocity_mps=tuple(float(v) for v in self._obs[_VEL]),
            yaw_rad=float(self._obs[_YAW]),
            battery_pct=self._battery_pct,
            is_flying=self._flying,
            sim_time_s=self._sim_time(),
        )

    def get_camera_image(self) -> np.ndarray:
        # TODO: downward-facing onboard camera for ground-plane object localisation.
        raise NotImplementedError("onboard camera is not available yet")

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    # ------------------------------------------------------------- sim-only API

    @property
    def control_period_s(self) -> float:
        """Duration of one control step (1 / control_hz), seconds."""
        return self._dt

    def add_step_callback(self, callback: Callable[[], None]) -> None:
        """Call `callback()` after every control step (e.g. to record frames)."""
        self._step_callbacks.append(callback)

    def render_frame(
        self,
        width: int = 640,
        height: int = 480,
        target_m: Vec3 | None = None,
        distance_m: float | None = None,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
    ) -> np.ndarray:
        """Render the scene from an external orbit camera, as an (H, W, 3) uint8 RGB array.

        The camera looks at `target_m` (default: the drone, i.e. a chase camera) from
        `distance_m` metres at the given yaw/pitch; unset values come from the GUI config.
        """
        g = self.config.gui
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(target_m) if target_m else self._obs[_POS].tolist(),
            distance=distance_m or g.camera_distance_m,
            yaw=g.camera_yaw_deg if yaw_deg is None else yaw_deg,
            pitch=g.camera_pitch_deg if pitch_deg is None else pitch_deg,
            roll=0.0,
            upAxisIndex=2,
            physicsClientId=self._client,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=width / height, nearVal=0.05, farVal=100.0
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=proj, physicsClientId=self._client
        )
        return np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]

    def add_marker(
        self,
        position_m: Vec3,
        radius_m: float = 0.01,
        rgba: tuple[float, float, float, float] = (1.0, 0.4, 0.0, 1.0),
    ) -> None:
        """Place a visual-only sphere (no collision, no mass), e.g. to draw a flight trail."""
        shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius_m, rgbaColor=list(rgba), physicsClientId=self._client
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shape,
            basePosition=list(position_m),
            physicsClientId=self._client,
        )

    # ------------------------------------------------------------------ internals

    def _move(
        self, start: np.ndarray, target: np.ndarray, speed_mps: float, verb: str
    ) -> MoveResult:
        if speed_mps <= 0:
            return self._fail(f"speed must be positive, got {speed_mps} m/s")
        t_start = self._start_command()
        self._ramp(start, target, speed_mps)
        reached = self._settle(target, self.config.motion.settle_timeout_s)
        if reached:
            return self._result(True, t_start, f"{verb} {_fmt(target)}")
        error = self._distance_to(target)
        return self._result(False, t_start, f"timed out {error:.2f} m from {_fmt(target)}")

    def _ramp(
        self,
        start: np.ndarray,
        target: np.ndarray,
        speed_mps: float,
        stop: Callable[[], bool] | None = None,
    ) -> None:
        """Slide the setpoint from `start` to `target` at `speed_mps`, with feed-forward."""
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance > 0:
            direction = delta / distance
            for k in range(1, math.ceil(distance / speed_mps / self._dt) + 1):
                travelled = min(k * self._dt * speed_mps, distance)
                velocity = direction * speed_mps if travelled < distance else np.zeros(3)
                self._step(start + direction * travelled, velocity)
                if stop is not None and stop():
                    break
        self._setpoint = target.copy()

    def _settle(
        self, target: np.ndarray, timeout_s: float, done: Callable[[], bool] | None = None
    ) -> bool:
        """Hold `target` until `done()` (default: within tolerance) or `timeout_s` of sim time."""
        if done is None:

            def done() -> bool:
                return self._distance_to(target) <= self.config.motion.position_tolerance_m

        for _ in range(math.ceil(timeout_s / self._dt)):
            if done():
                return True
            self._step(target, np.zeros(3))
        return done()

    def _run_steps(self, n: int, setpoint: np.ndarray | None = None) -> None:
        for _ in range(n):
            self._step(setpoint, np.zeros(3))

    def _step(self, setpoint: np.ndarray | None, velocity: np.ndarray) -> None:
        """Advance one control step. Motors are off when not flying."""
        if self._flying and setpoint is not None:
            rpm, _, _ = self._ctrl.computeControlFromState(
                control_timestep=self._dt, state=self._obs, target_pos=setpoint, target_vel=velocity
            )
        else:
            rpm = np.zeros(4)
        obs, *_ = self._env.step(rpm.reshape(1, 4))
        self._obs = obs[0]
        if self._flying:
            self._drain_battery()
        for callback in self._step_callbacks:
            callback()
        if self._gui and self.config.gui.realtime:
            self._sync_wall_clock()

    def _drain_battery(self) -> None:
        b = self.config.battery
        horizontal_speed = float(np.linalg.norm(self._obs[_VEL][:2]))
        rate = b.hover_drain_pct_per_s + b.speed_drain_pct_per_mps * horizontal_speed
        self._battery_pct = max(0.0, self._battery_pct - rate * self._dt)

    def _sync_wall_clock(self) -> None:
        wall0, sim0 = self._wall_anchor
        ahead_s = (self._sim_time() - sim0) - (time.monotonic() - wall0)
        if ahead_s > 0:
            time.sleep(ahead_s)

    def _start_command(self) -> float:
        now = self._sim_time()
        self._wall_anchor = (time.monotonic(), now)
        return now

    def _result(self, success: bool, t_start: float, message: str) -> MoveResult:
        elapsed = self._sim_time() - t_start
        result = MoveResult(
            success, self.get_state().position_m, elapsed, f"{message} in {elapsed:.1f} s"
        )
        logger.info("%s: %s", "ok" if success else "FAILED", result.message)
        return result

    def _fail(self, message: str) -> MoveResult:
        logger.info("FAILED: %s", message)
        return MoveResult(False, self.get_state().position_m, 0.0, message)

    def _distance_to(self, target: np.ndarray) -> float:
        return float(np.linalg.norm(self._obs[_POS] - target))

    def _raw_sim_time(self) -> float:
        return self._env.step_counter * self._env.PYB_TIMESTEP

    def _sim_time(self) -> float:
        return self._raw_sim_time() - self._t0
