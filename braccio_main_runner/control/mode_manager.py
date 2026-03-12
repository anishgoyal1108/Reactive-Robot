"""Control mode state machine for nominal/obstacle/fallback/comms-fault."""

from __future__ import annotations

from dataclasses import dataclass


MODES = {
    "nominal_tracking",
    "obstacle_aware_tracking",
    "fallback_hold",
    "comms_fault",
}


@dataclass
class ModeStatus:
    mode: str = "nominal_tracking"
    optimizer_active: bool = False
    fallback_active: bool = False


class ModeManager:
    def __init__(self):
        self.status = ModeStatus()

    def set_nominal(self) -> None:
        self.status.mode = "nominal_tracking"
        self.status.optimizer_active = False
        self.status.fallback_active = False

    def set_obstacle(self) -> None:
        self.status.mode = "obstacle_aware_tracking"
        self.status.optimizer_active = True
        self.status.fallback_active = False

    def set_fallback(self) -> None:
        self.status.mode = "fallback_hold"
        self.status.optimizer_active = False
        self.status.fallback_active = True

    def set_comms_fault(self) -> None:
        self.status.mode = "comms_fault"
        self.status.optimizer_active = False
        self.status.fallback_active = True
