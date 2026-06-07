"""Application configuration for the gesture control subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VideoConfig:
    """Video capture settings."""

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    mirror_frame: bool = True


@dataclass(frozen=True)
class HandDetectionConfig:
    """MediaPipe hand detector settings."""

    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class StaticClassifierConfig:
    """Rule-based static classifier thresholds."""

    extended_finger_ratio: float = 1.12
    thumb_extended_ratio: float = 1.05
    thumb_tip_extension_ratio: float = 0.55
    thumb_vertical_clearance_ratio: float = 0.45
    direction_margin: float = 0.05
    ok_tip_distance_ratio: float = 0.28
    min_confidence: float = 0.6


@dataclass(frozen=True)
class DynamicClassifierConfig:
    """Trajectory-based dynamic classifier thresholds."""

    buffer_size: int = 30
    min_window_points: int = 5
    missing_detection_tolerance_frames: int = 8
    min_horizontal_displacement: float = 0.18
    max_vertical_drift: float = 0.14
    min_wave_direction_changes: int = 2
    min_circle_radius: float = 0.04
    max_circle_radius_cv: float = 0.65
    min_circle_angle_span: float = 4.5
    max_circle_angle_span: float = 7.4
    min_pull_scale_growth: float = 0.10
    min_confidence: float = 0.6
    selection_min_confidence: float = 0.30
    motion_start_threshold: float = 0.02
    motion_end_threshold: float = 0.012
    motion_start_frames: int = 2
    motion_end_frames: int = 3
    min_dynamic_segment_points: int = 5
    max_dynamic_segment_points: int = 36
    early_dynamic_min_points: int = 10
    early_dynamic_min_confidence: float = 0.85
    static_priority_min_confidence: float = 0.65
    static_priority_frames: int = 2
    static_transition_grace_frames: int = 4
    dynamic_min_confirm_points: int = 5
    dynamic_min_confirm_path: float = 0.08
    dynamic_min_confirm_mean_step: float = 0.006
    dynamic_min_confirm_scale_growth: float = 0.08
    dynamic_static_compatibility_min_confidence: float = 0.65
    min_pull_confirm_sustained_growth: float = 0.12
    min_pull_confirm_positive_step_ratio: float = 0.55


@dataclass(frozen=True)
class CalibrationConfig:
    """Adaptive user calibration settings."""

    samples_per_gesture: int = 5
    sigma_multiplier: float = 2.0
    min_calibrated_confidence: float = 0.65
    profile_directory: str = "data/user_profiles"


@dataclass(frozen=True)
class CommandMappingConfig:
    """Debouncing and command emission settings."""

    static_confirmation_frames: int = 5
    dynamic_confirmation_frames: int = 1
    emergency_confirmation_frames: int = 3
    min_confidence: float = 0.65
    dynamic_min_confidence: float = 0.30
    emergency_min_confidence: float = 0.85
    repeat_same_command: bool = False
    suppress_static_commands_during_motion: bool = True


@dataclass(frozen=True)
class SenderConfig:
    """Robot command transport settings."""

    serial_port: str = "COM3"
    baudrate: int = 115200
    timeout_seconds: float = 1.0
    ros_topic: str = "/cmd_vel"


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object used by the subsystem."""

    video: VideoConfig = field(default_factory=VideoConfig)
    hand_detection: HandDetectionConfig = field(default_factory=HandDetectionConfig)
    static_classifier: StaticClassifierConfig = field(default_factory=StaticClassifierConfig)
    dynamic_classifier: DynamicClassifierConfig = field(default_factory=DynamicClassifierConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    command_mapping: CommandMappingConfig = field(default_factory=CommandMappingConfig)
    sender: SenderConfig = field(default_factory=SenderConfig)
