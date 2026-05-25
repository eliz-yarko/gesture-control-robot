"""Map recognized gestures to robot commands with debouncing."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import CommandMappingConfig
from src.domain import CommandEvent, GestureID, GesturePrediction, RobotCommand

DEFAULT_GESTURE_COMMAND_MAP: dict[GestureID, RobotCommand] = {
    GestureID.OPEN_PALM: RobotCommand.STOP,
    GestureID.FIST: RobotCommand.FORWARD,
    GestureID.THUMB_UP: RobotCommand.START,
    GestureID.THUMB_DOWN: RobotCommand.EMERGENCY_STOP,
    GestureID.INDEX_LEFT: RobotCommand.TURN_LEFT,
    GestureID.INDEX_RIGHT: RobotCommand.TURN_RIGHT,
    GestureID.PEACE: RobotCommand.INCREASE_SPEED,
    GestureID.THREE_FINGERS: RobotCommand.DECREASE_SPEED,
    GestureID.PINKY: RobotCommand.RETURN_HOME,
    GestureID.OK_SIGN: RobotCommand.CONFIRM_ACTION,
    GestureID.WAVE_LR: RobotCommand.MODE_TOGGLE,
    GestureID.CIRCLE: RobotCommand.ROTATE_360,
    GestureID.PULL_TOWARD: RobotCommand.APPROACH_OPERATOR,
}


@dataclass
class _DebounceState:
    gesture_id: GestureID = GestureID.UNKNOWN
    frames: int = 0


class CommandMapper:
    """Convert gesture predictions into confirmed robot commands."""

    def __init__(
        self,
        config: CommandMappingConfig | None = None,
        gesture_command_map: dict[GestureID, RobotCommand] | None = None,
    ) -> None:
        """Initialize command mapper."""

        self._config = config or CommandMappingConfig()
        self._gesture_command_map = gesture_command_map or DEFAULT_GESTURE_COMMAND_MAP
        self._state = _DebounceState()
        self._last_emitted: RobotCommand | None = None

    def reset(self) -> None:
        """Reset debouncing state and emitted-command memory."""

        self._state = _DebounceState()
        self._last_emitted = None

    def update(self, prediction: GesturePrediction) -> CommandEvent | None:
        """Update mapper state and return a command once it is confirmed."""

        if (
            prediction.gesture_id == GestureID.UNKNOWN
            or prediction.confidence < self._config.min_confidence
        ):
            self._state = _DebounceState()
            return None

        command = self._gesture_command_map.get(prediction.gesture_id)
        if command is None:
            self._state = _DebounceState()
            return None

        if prediction.gesture_id == self._state.gesture_id:
            self._state.frames += 1
        else:
            self._state = _DebounceState(gesture_id=prediction.gesture_id, frames=1)

        if self._state.frames < self._required_frames(prediction.gesture_id):
            return None

        if not self._config.repeat_same_command and self._last_emitted == command:
            return None

        self._last_emitted = command
        return CommandEvent(
            command=command,
            gesture_id=prediction.gesture_id,
            confidence=prediction.confidence,
        )

    def _required_frames(self, gesture_id: GestureID) -> int:
        if gesture_id == GestureID.THUMB_DOWN:
            return self._config.emergency_confirmation_frames
        if gesture_id.is_dynamic:
            return self._config.dynamic_confirmation_frames
        return self._config.static_confirmation_frames
