"""JSON-serializable user calibration profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import CalibrationConfig
from src.domain import GestureID


@dataclass(frozen=True)
class GestureCalibrationStats:
    """Personalized statistics for one gesture."""

    gesture_id: GestureID
    sample_count: int
    centroid: tuple[float, ...]
    std: tuple[float, ...]
    radius: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible representation."""

        return {
            "gesture_id": int(self.gesture_id),
            "sample_count": self.sample_count,
            "centroid": list(self.centroid),
            "std": list(self.std),
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GestureCalibrationStats:
        """Build stats from JSON-compatible representation."""

        return cls(
            gesture_id=GestureID(int(data["gesture_id"])),
            sample_count=int(data["sample_count"]),
            centroid=tuple(float(value) for value in data["centroid"]),
            std=tuple(float(value) for value in data["std"]),
            radius=float(data["radius"]),
        )


@dataclass(frozen=True)
class UserCalibrationProfile:
    """Calibration profile for one operator."""

    user_id: str
    created_at: str
    updated_at: str
    feature_size: int
    samples_per_gesture: int
    sigma_multiplier: float
    gestures: dict[GestureID, GestureCalibrationStats]

    def stats_for(self, gesture_id: GestureID) -> GestureCalibrationStats | None:
        """Return calibration statistics for a gesture if present."""

        return self.gestures.get(gesture_id)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible representation."""

        return {
            "schema_version": 1,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "feature_size": self.feature_size,
            "samples_per_gesture": self.samples_per_gesture,
            "sigma_multiplier": self.sigma_multiplier,
            "gestures": {
                str(int(gesture_id)): stats.to_dict()
                for gesture_id, stats in sorted(
                    self.gestures.items(),
                    key=lambda item: int(item[0]),
                )
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserCalibrationProfile:
        """Build a profile from JSON-compatible representation."""

        gestures = {
            GestureID(int(gesture_id)): GestureCalibrationStats.from_dict(stats)
            for gesture_id, stats in data["gestures"].items()
        }
        return cls(
            user_id=str(data["user_id"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            feature_size=int(data["feature_size"]),
            samples_per_gesture=int(data["samples_per_gesture"]),
            sigma_multiplier=float(data["sigma_multiplier"]),
            gestures=gestures,
        )


class CalibrationProfileStore:
    """Persist calibration profiles as JSON files."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        """Initialize the store."""

        self._config = config or CalibrationConfig()
        self._directory = Path(self._config.profile_directory)

    @property
    def directory(self) -> Path:
        """Return profile directory."""

        return self._directory

    def save(self, profile: UserCalibrationProfile) -> Path:
        """Save a user calibration profile and return its path."""

        self._directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(profile.user_id)
        path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def load(self, user_id: str) -> UserCalibrationProfile:
        """Load a profile by user ID."""

        path = self.path_for(user_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserCalibrationProfile.from_dict(data)

    def load_path(self, path: str | Path) -> UserCalibrationProfile:
        """Load a profile from an explicit path."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return UserCalibrationProfile.from_dict(data)

    def path_for(self, user_id: str) -> Path:
        """Return the profile path for a user ID."""

        safe_user_id = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in user_id
        )
        return self._directory / f"{safe_user_id}.json"


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()
