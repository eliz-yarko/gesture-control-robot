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


@dataclass(frozen=True)
class CommandConfirmationState:
    """Current command candidate and debounce progress."""

    gesture_id: GestureID = GestureID.UNKNOWN
    command: RobotCommand = RobotCommand.UNKNOWN
    confidence: float = 0.0
    stable_frames: int = 0
    required_frames: int = 0
    ready: bool = False
    blocked_reason: str = "unknown"

    @classmethod
    def unknown(cls, reason: str = "unknown") -> CommandConfirmationState:
        """Build an empty confirmation state."""

        return cls(blocked_reason=reason)


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
        self._confirmation_state = CommandConfirmationState.unknown()

    def reset(self) -> None:
        """Reset debouncing state and emitted-command memory."""

        self._state = _DebounceState()
        self._last_emitted = None
        self._confirmation_state = CommandConfirmationState.unknown("reset")

    @property
    def confirmation_state(self) -> CommandConfirmationState:
        """Return the latest command confirmation progress."""

        return self._confirmation_state

    def configure(self, config: CommandMappingConfig) -> None:
        """Update confirmation thresholds used for future predictions."""

        if config == self._config:
            return
        self._config = config
        self._state = _DebounceState()
        self._confirmation_state = CommandConfirmationState.unknown("settings_updated")

    def update(self, prediction: GesturePrediction) -> CommandEvent | None:
        """Update mapper state and return a command once it is confirmed."""

        if prediction.gesture_id == GestureID.UNKNOWN:
            self._state = _DebounceState()
            self._last_emitted = None
            self._confirmation_state = CommandConfirmationState.unknown(
                prediction.metadata.get("reason", "unknown_prediction")
            )
            return None

        command = self._gesture_command_map.get(prediction.gesture_id, RobotCommand.UNKNOWN)
        required_frames = self._required_frames(prediction.gesture_id)
        if prediction.confidence < self._config.min_confidence:
            self._state = _DebounceState()
            self._confirmation_state = CommandConfirmationState(
                gesture_id=prediction.gesture_id,
                command=command,
                confidence=prediction.confidence,
                stable_frames=0,
                required_frames=required_frames,
                ready=False,
                blocked_reason="confidence_below_threshold",
            )
            return None

        if command == RobotCommand.UNKNOWN:
            self._state = _DebounceState()
            self._confirmation_state = CommandConfirmationState(
                gesture_id=prediction.gesture_id,
                command=RobotCommand.UNKNOWN,
                confidence=prediction.confidence,
                stable_frames=0,
                required_frames=required_frames,
                ready=False,
                blocked_reason="unmapped_gesture",
            )
            return None

        if prediction.gesture_id == self._state.gesture_id:
            self._state.frames += 1
        else:
            self._state = _DebounceState(gesture_id=prediction.gesture_id, frames=1)

        ready = self._state.frames >= required_frames
        self._confirmation_state = CommandConfirmationState(
            gesture_id=prediction.gesture_id,
            command=command,
            confidence=prediction.confidence,
            stable_frames=self._state.frames,
            required_frames=required_frames,
            ready=ready,
            blocked_reason="" if ready else "debouncing",
        )

        if not ready:
            return None

        if not self._config.repeat_same_command and self._last_emitted == command:
            self._confirmation_state = CommandConfirmationState(
                gesture_id=prediction.gesture_id,
                command=command,
                confidence=prediction.confidence,
                stable_frames=self._state.frames,
                required_frames=required_frames,
                ready=True,
                blocked_reason="repeat_suppressed",
            )
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
