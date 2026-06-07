"""Gesture recognition package."""

from src.recognition.dynamic_classifier import DynamicGestureClassifier
from src.recognition.dynamic_segmenter import DynamicGestureSegmenter
from src.recognition.gesture_pose import (
    EXPECTED_STATIC_POSES,
    FINGER_NAMES,
    GesturePoseSpec,
    StaticPoseAnalysis,
    analyze_static_pose,
    expected_pose_for,
)
from src.recognition.hand_detector import HandDetection, HandDetector
from src.recognition.model_classifier import (
    ConfidenceFallbackStaticGestureClassifier,
    FallbackDynamicGestureClassifier,
    FallbackStaticGestureClassifier,
    SklearnDynamicGestureClassifier,
    SklearnStaticGestureClassifier,
    load_threshold_profile,
)
from src.recognition.static_classifier import FingerStates, StaticGestureClassifier
from src.recognition.trajectory_buffer import TrajectoryBuffer, TrajectoryPoint

__all__ = [
    "DynamicGestureClassifier",
    "DynamicGestureSegmenter",
    "EXPECTED_STATIC_POSES",
    "FINGER_NAMES",
    "ConfidenceFallbackStaticGestureClassifier",
    "FallbackDynamicGestureClassifier",
    "FallbackStaticGestureClassifier",
    "FingerStates",
    "GesturePoseSpec",
    "HandDetection",
    "HandDetector",
    "SklearnDynamicGestureClassifier",
    "SklearnStaticGestureClassifier",
    "StaticGestureClassifier",
    "TrajectoryBuffer",
    "TrajectoryPoint",
    "StaticPoseAnalysis",
    "analyze_static_pose",
    "expected_pose_for",
    "load_threshold_profile",
]
