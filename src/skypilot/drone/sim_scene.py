"""PyBullet scene helpers for SimBackend: world objects, markers and external rendering."""

from __future__ import annotations

import numpy as np
import pybullet as p

from skypilot.config import WorldConfig
from skypilot.drone.backend import Vec3

RGBA = tuple[float, float, float, float]


def spawn_box(client: int, centre_m: Vec3, size_m: float, rgba: RGBA) -> int:
    """Add a static, solid cube of side `size_m` centred at `centre_m`. Returns its body id."""
    half = [size_m / 2] * 3
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half, physicsClientId=client
        ),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=list(rgba), physicsClientId=client
        ),
        basePosition=list(centre_m),
        physicsClientId=client,
    )


def spawn_world(client: int, world: WorldConfig) -> dict[str, int]:
    """Place every object of `world` on the ground. Returns body ids by object name."""
    return {
        obj.name: spawn_box(
            client,
            (obj.position_m[0], obj.position_m[1], obj.size_m / 2),
            obj.size_m,
            world.colours[obj.colour],
        )
        for obj in world.objects
    }


def add_marker(client: int, position_m: Vec3, radius_m: float, rgba: RGBA) -> int:
    """Add a visual-only sphere (no collision, no mass). Returns its body id."""
    shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius_m, rgbaColor=list(rgba), physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=shape,
        basePosition=list(position_m),
        physicsClientId=client,
    )


def render_orbit(
    client: int,
    width: int,
    height: int,
    target_m: Vec3,
    distance_m: float,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    """Render from an external camera orbiting `target_m`, as an (H, W, 3) uint8 RGB array."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(target_m),
        distance=distance_m,
        yaw=yaw_deg,
        pitch=pitch_deg,
        roll=0.0,
        upAxisIndex=2,
        physicsClientId=client,
    )
    proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=width / height, nearVal=0.05, farVal=100.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, viewMatrix=view, projectionMatrix=proj, physicsClientId=client
    )
    return np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]
