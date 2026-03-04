"""
state_library.py — Named pose snapshots, persisted to JSON.

A "state" captures the full arm pose:
  - joints:        6 servo angles (the exact angles sent to hardware)
  - theta/r/z:     IK polar coordinates (restores the IK display on recall)
  - wrist_offset, wrist_rot, gripper: remaining IK parameters

States are stored in a JSON file next to this module so they persist
across sessions.
"""

import json
import os
from typing import Optional

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'states.json')


class StateLibrary:
    """
    Dict-backed store of named arm states, persisted to a JSON file.

    Thread-safety: all public methods are called from the curses main
    thread only, so no extra locking is needed here.
    """

    def __init__(self, path: str = _DEFAULT_PATH):
        self._path   = path
        self._states: dict = {}   # name → state dict (insertion order)
        self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def names(self) -> list:
        """Return state names in insertion order."""
        return list(self._states.keys())

    def save_state(self, name: str, snapshot: dict) -> None:
        """
        Store a named state from an ArmState.snapshot() dict.
        Only the fields needed to recreate the pose are kept.
        Overwrites any existing state with the same name.
        """
        self._states[name] = {
            'joints':       list(snapshot['joints']),
            'theta':        snapshot['theta'],
            'r':            snapshot['r'],
            'z':            snapshot['z'],
            'wrist_offset': snapshot['wrist_offset'],
            'wrist_rot':    snapshot['wrist_rot'],
            'gripper':      snapshot['gripper'],
        }
        self._persist()

    def get_state(self, name: str) -> Optional[dict]:
        """Return the state dict for name, or None if not found."""
        return self._states.get(name)

    def delete_state(self, name: str) -> bool:
        """Delete a state by name.  Returns True if it existed."""
        if name in self._states:
            del self._states[name]
            self._persist()
            return True
        return False

    def rename_state(self, old: str, new: str) -> bool:
        """Rename a state key, preserving insertion order."""
        if old not in self._states or not new:
            return False
        # Rebuild dict to preserve ordering without changing values
        self._states = {
            (new if k == old else k): v
            for k, v in self._states.items()
        }
        self._persist()
        return True

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(self._path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._states = data
        except (FileNotFoundError, json.JSONDecodeError):
            self._states = {}

    def _persist(self) -> None:
        try:
            with open(self._path, 'w') as f:
                json.dump(self._states, f, indent=2)
        except OSError:
            pass
