from __future__ import annotations

from src.config import CommandMappingConfig
from src.domain import GestureID, GesturePrediction, RobotCommand
from src.interpretation.command_mapper import CommandMapper


def test_command_mapper_emits_static_command_after_debounce() -> None:
    mapper = CommandMapper(CommandMappingConfig(static_confirmation_frames=3))
    prediction = GesturePrediction(GestureID.OPEN_PALM, 0.9)

    assert mapper.update(prediction) is None
    assert mapper.update(prediction) is None
    event = mapper.update(prediction)

    assert event is not None
    assert event.command == RobotCommand.STOP
    assert event.gesture_id == GestureID.OPEN_PALM


def test_command_mapper_uses_shorter_emergency_debounce() -> None:
    mapper = CommandMapper(
        CommandMappingConfig(
            static_confirmation_frames=5,
            emergency_confirmation_frames=2,
        )
    )
    prediction = GesturePrediction(GestureID.THUMB_DOWN, 0.95)

    assert mapper.update(prediction) is None
    event = mapper.update(prediction)

    assert event is not None
    assert event.command == RobotCommand.EMERGENCY_STOP


def test_command_mapper_emits_dynamic_command_immediately_by_default() -> None:
    mapper = CommandMapper()

    event = mapper.update(GesturePrediction(GestureID.CIRCLE, 0.98))

    assert event is not None
    assert event.command == RobotCommand.ROTATE_360
    assert event.gesture_id == GestureID.CIRCLE


def test_command_mapper_resets_on_unknown_prediction() -> None:
    mapper = CommandMapper(CommandMappingConfig(static_confirmation_frames=2))

    assert mapper.update(GesturePrediction(GestureID.FIST, 0.9)) is None
    assert mapper.update(GesturePrediction.unknown()) is None
    assert mapper.update(GesturePrediction(GestureID.FIST, 0.9)) is None
    event = mapper.update(GesturePrediction(GestureID.FIST, 0.9))

    assert event is not None
    assert event.command == RobotCommand.FORWARD


def test_command_mapper_does_not_repeat_same_command_by_default() -> None:
    mapper = CommandMapper(CommandMappingConfig(static_confirmation_frames=1))
    prediction = GesturePrediction(GestureID.OPEN_PALM, 0.9)

    first_event = mapper.update(prediction)
    second_event = mapper.update(prediction)

    assert first_event is not None
    assert second_event is None


def test_command_mapper_reports_confirmation_progress() -> None:
    mapper = CommandMapper(CommandMappingConfig(static_confirmation_frames=3))
    prediction = GesturePrediction(GestureID.OPEN_PALM, 0.9)

    assert mapper.update(prediction) is None

    state = mapper.confirmation_state
    assert state.gesture_id == GestureID.OPEN_PALM
    assert state.command == RobotCommand.STOP
    assert state.stable_frames == 1
    assert state.required_frames == 3
    assert state.ready is False
    assert state.blocked_reason == "debouncing"


def test_command_mapper_reports_low_confidence_block() -> None:
    mapper = CommandMapper(CommandMappingConfig(min_confidence=0.65))

    assert mapper.update(GesturePrediction(GestureID.THUMB_DOWN, 0.31)) is None

    state = mapper.confirmation_state
    assert state.gesture_id == GestureID.THUMB_DOWN
    assert state.command == RobotCommand.EMERGENCY_STOP
    assert state.stable_frames == 0
    assert state.blocked_reason == "confidence_below_threshold"
