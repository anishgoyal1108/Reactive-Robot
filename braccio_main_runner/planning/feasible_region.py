"""Feasible workspace cloud generation and obstacle-region membership checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from planning.urdf_kinematics import BraccioURDFKinematics

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None


@dataclass(frozen=True)
class FeasibleRegionConfig:
    sample_count: int = 30000
    seed: int = 7
    membership_tol_m: float = 0.07


class FeasibleRegion:
    def __init__(self, points_m: np.ndarray, membership_tol_m: float = 0.07):
        pts = np.asarray(points_m, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_m must be Nx3")
        self.points_m = pts
        self.membership_tol_m = float(membership_tol_m)
        self._tree = KDTree(self.points_m) if KDTree is not None and self.points_m.size > 0 else None

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.min(self.points_m, axis=0), np.max(self.points_m, axis=0)

    def nearest_distance(self, point_m: np.ndarray) -> float:
        p = np.asarray(point_m, dtype=float).reshape(3)
        if self.points_m.size == 0:
            return float("inf")
        if self._tree is not None:
            d, _ = self._tree.query(p, k=1)
            return float(d)
        return float(np.min(np.linalg.norm(self.points_m - p[None, :], axis=1)))

    def contains(self, point_m: np.ndarray, tol_m: float | None = None) -> bool:
        tol = self.membership_tol_m if tol_m is None else float(tol_m)
        return self.nearest_distance(point_m) <= tol


class FeasibleRegionBuilder:
    def __init__(self, urdf_path: str | Path, output_dir: str | Path):
        self.urdf_path = Path(urdf_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_or_build(self, config: FeasibleRegionConfig) -> FeasibleRegion:
        cloud_path = self.output_dir / "workspace_cloud.npy"
        meta_path = self.output_dir / "workspace_meta.json"

        if cloud_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if int(meta.get("sample_count", -1)) == int(config.sample_count):
                    pts = np.load(cloud_path)
                    region = FeasibleRegion(pts, membership_tol_m=config.membership_tol_m)
                    self._save_plane_plots(region.points_m)
                    return region
            except Exception:
                pass

        pts = self._sample_workspace_cloud(config)
        np.save(cloud_path, pts)

        lo = np.min(pts, axis=0)
        hi = np.max(pts, axis=0)
        meta = {
            "urdf": str(self.urdf_path),
            "sample_count": int(config.sample_count),
            "seed": int(config.seed),
            "bounds_min_m": [float(lo[0]), float(lo[1]), float(lo[2])],
            "bounds_max_m": [float(hi[0]), float(hi[1]), float(hi[2])],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self._save_plane_plots(pts)
        return FeasibleRegion(pts, membership_tol_m=config.membership_tol_m)

    def _sample_workspace_cloud(self, config: FeasibleRegionConfig) -> np.ndarray:
        kin = BraccioURDFKinematics(self.urdf_path)
        rng = np.random.default_rng(int(config.seed))

        n = int(config.sample_count)
        base = rng.uniform(0.0, 180.0, size=n)
        shoulder = rng.uniform(15.0, 165.0, size=n)
        elbow = rng.uniform(0.0, 180.0, size=n)
        wrist_v = rng.uniform(0.0, 180.0, size=n)
        wrist_r = rng.uniform(0.0, 180.0, size=n)
        gripper = rng.uniform(10.0, 73.0, size=n)

        pts = np.zeros((n, 3), dtype=float)
        for i in range(n):
            joints = [
                float(base[i]),
                float(shoulder[i]),
                float(elbow[i]),
                float(wrist_v[i]),
                float(wrist_r[i]),
                float(gripper[i]),
            ]
            pts[i, :] = kin.end_effector_position_m(joints)

        # Keep only points in front hemisphere (matching practical arm operation envelope).
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        return pts

    def _save_plane_plots(self, points_m: np.ndarray) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._plot_plane(points_m, 0, 1, "x [m]", "y [m]", self.output_dir / "feasible_xy.png")
        self._plot_plane(points_m, 1, 2, "y [m]", "z [m]", self.output_dir / "feasible_yz.png")
        self._plot_plane(points_m, 2, 0, "z [m]", "x [m]", self.output_dir / "feasible_zx.png")

    @staticmethod
    def _plot_plane(points_m: np.ndarray, i: int, j: int, xl: str, yl: str, out_path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=120)
        ax.scatter(points_m[:, i], points_m[:, j], s=2, c="#8fd3ff", alpha=0.18, linewidths=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title("Feasible Workspace Projection")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
