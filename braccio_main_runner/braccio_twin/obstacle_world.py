"""
obstacle_world.py — User-facing obstacle management for the sim.

The block-based editor (future phase) and the CLI (`--world world.json`)
both call into ObstacleWorld to spawn and remove obstacles. This is
deliberately a thin wrapper over PyBullet's createMultiBody — we only
track the body IDs so we can remove them later and serialise the scene
back out.

ObstacleSpec captures everything needed to round-trip an obstacle
through JSON. Supported shapes: "box", "sphere", "cylinder". Everything
is in mm at the API level (consistent with the rest of the control
stack) but gets converted to metres before it reaches PyBullet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from .sim_arm import SimArm


ShapeName = Literal["box", "sphere", "cylinder"]


@dataclass
class ObstacleSpec:
    """
    Declarative description of one obstacle.

    Fields
    ------
    name       : unique string id — used for removal and logging
    shape      : "box" | "sphere" | "cylinder"
    position_mm: (x, y, z) centre, in arm-base frame, millimetres
    size_mm    : shape-dependent dimensions in mm
                 - box:      (width, depth, height)
                 - sphere:   (radius, _, _)   (only first element used)
                 - cylinder: (radius, _, height)
    rgba       : colour for the visual, RGBA 0..1
    """
    name: str
    shape: ShapeName = "box"
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_mm: tuple[float, float, float] = (40.0, 40.0, 40.0)
    rgba: tuple[float, float, float, float] = (0.85, 0.15, 0.15, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObstacleSpec":
        return cls(
            name=d["name"],
            shape=d.get("shape", "box"),
            position_mm=tuple(d.get("position_mm", (0.0, 0.0, 0.0))),
            size_mm=tuple(d.get("size_mm", (40.0, 40.0, 40.0))),
            rgba=tuple(d.get("rgba", (0.85, 0.15, 0.15, 1.0))),
        )


class ObstacleWorld:
    """Manages spawned obstacles and their PyBullet body handles."""

    def __init__(self, sim_arm: SimArm):
        self._arm = sim_arm
        self._pb = sim_arm._pybullet
        self._client = sim_arm.get_client_id()
        # name → (body_id, ObstacleSpec)
        self._bodies: dict[str, tuple[int, ObstacleSpec]] = {}

    # ── spawn / despawn ─────────────────────────────────────────────────

    def add(self, spec: ObstacleSpec) -> int:
        """
        Spawn an obstacle. Returns the PyBullet body id.
        If `spec.name` already exists, the previous body is removed first.
        """
        if spec.name in self._bodies:
            self.remove(spec.name)

        pb = self._pb
        pos_m = tuple(v / 1000.0 for v in spec.position_mm)
        rgba = list(spec.rgba)

        if spec.shape == "box":
            half = [max(1e-3, v / 2000.0) for v in spec.size_mm]
            col = pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=half, physicsClientId=self._client
            )
            vis = pb.createVisualShape(
                pb.GEOM_BOX,
                halfExtents=half,
                rgbaColor=rgba,
                physicsClientId=self._client,
            )
        elif spec.shape == "sphere":
            radius = max(1e-3, spec.size_mm[0] / 1000.0)
            col = pb.createCollisionShape(
                pb.GEOM_SPHERE, radius=radius, physicsClientId=self._client
            )
            vis = pb.createVisualShape(
                pb.GEOM_SPHERE,
                radius=radius,
                rgbaColor=rgba,
                physicsClientId=self._client,
            )
        elif spec.shape == "cylinder":
            radius = max(1e-3, spec.size_mm[0] / 1000.0)
            height = max(1e-3, spec.size_mm[2] / 1000.0)
            col = pb.createCollisionShape(
                pb.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=self._client,
            )
            vis = pb.createVisualShape(
                pb.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=rgba,
                physicsClientId=self._client,
            )
        else:
            raise ValueError(f"unknown obstacle shape: {spec.shape!r}")

        body_id = pb.createMultiBody(
            baseMass=0.0,             # static obstacle
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(pos_m),
            physicsClientId=self._client,
        )
        self._bodies[spec.name] = (body_id, spec)
        return body_id

    def remove(self, name: str) -> bool:
        """Remove an obstacle by name. Returns True if it existed."""
        entry = self._bodies.pop(name, None)
        if entry is None:
            return False
        body_id, _spec = entry
        try:
            self._pb.removeBody(body_id, physicsClientId=self._client)
        except Exception:
            pass
        return True

    def clear(self) -> None:
        """Remove every obstacle."""
        for name in list(self._bodies.keys()):
            self.remove(name)

    def list(self) -> list[ObstacleSpec]:
        """Return a list of the current ObstacleSpecs."""
        return [spec for (_bid, spec) in self._bodies.values()]

    # ── JSON round-trip ─────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {"obstacles": [spec.to_dict() for spec in self.list()]}
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> int:
        """Replace current world with contents of a JSON file. Returns count added."""
        data = json.loads(Path(path).read_text())
        self.clear()
        added = 0
        for entry in data.get("obstacles", []):
            try:
                self.add(ObstacleSpec.from_dict(entry))
                added += 1
            except Exception:
                continue
        return added
