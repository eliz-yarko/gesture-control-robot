from __future__ import annotations

from src.domain import GestureID
from src.recognition.landmark_features import (
    DYNAMIC_FEATURE_VERSION,
    STATIC_FEATURE_VERSION,
    extract_static_features,
    extract_trajectory_features,
)
from src.recognition.model_classifier import (
    ModelBundle,
    SklearnDynamicGestureClassifier,
    SklearnStaticGestureClassifier,
)
from src.recognition.trajectory_buffer import TrajectoryBuffer, TrajectoryPoint


def test_static_model_classifier_uses_highest_probability_label() -> None:
    classifier = SklearnStaticGestureClassifier(
        ModelBundle(
            model=_FakeModel(("FIST", "OPEN_PALM"), (0.12, 0.88)),
            label_names=("FIST", "OPEN_PALM"),
            feature_version=STATIC_FEATURE_VERSION,
            min_confidence=0.2,
            model_type="fake",
        )
    )

    prediction = classifier.classify(_open_palm_landmarks())

    assert prediction.gesture_id == GestureID.OPEN_PALM
    assert prediction.confidence == 0.88


def test_static_model_classifier_rejects_low_confidence_prediction() -> None:
    classifier = SklearnStaticGestureClassifier(
        ModelBundle(
            model=_FakeModel(("FIST", "OPEN_PALM"), (0.51, 0.49)),
            label_names=("FIST", "OPEN_PALM"),
            feature_version=STATIC_FEATURE_VERSION,
            min_confidence=0.7,
            model_type="fake",
        )
    )

    prediction = classifier.classify(_open_palm_landmarks())

    assert prediction.gesture_id == GestureID.UNKNOWN


def test_dynamic_model_classifier_uses_trajectory_features() -> None:
    classifier = SklearnDynamicGestureClassifier(
        ModelBundle(
            model=_FakeModel(("CIRCLE", "WAVE_LR"), (0.25, 0.75)),
            label_names=("CIRCLE", "WAVE_LR"),
            feature_version=DYNAMIC_FEATURE_VERSION,
            min_confidence=0.3,
            model_type="fake",
        ),
        min_points=2,
    )
    buffer = TrajectoryBuffer(max_size=30)
    for index, x in enumerate((0.2, 0.6, 0.3)):
        buffer.add_point(
            TrajectoryPoint(
                palm_center=(x, 0.4, 0.0),
                index_tip=(x, 0.2, 0.0),
                hand_size=0.2,
                timestamp=float(index),
            )
        )

    prediction = classifier.classify(buffer)

    assert prediction.gesture_id == GestureID.WAVE_LR


def test_feature_extractors_return_stable_lengths() -> None:
    static_features = extract_static_features(_open_palm_landmarks())
    dynamic_features = extract_trajectory_features(
        [
            TrajectoryPoint(
                palm_center=(0.2, 0.4, 0.0),
                index_tip=(0.2, 0.2, 0.0),
                hand_size=0.2,
                timestamp=0.0,
            ),
            TrajectoryPoint(
                palm_center=(0.6, 0.4, 0.0),
                index_tip=(0.6, 0.2, 0.0),
                hand_size=0.3,
                timestamp=1.0,
            ),
        ]
    )

    assert len(static_features) == 98
    assert len(dynamic_features) == 32


class _FakeModel:
    def __init__(self, classes: tuple[str, ...], probabilities: tuple[float, ...]) -> None:
        self.classes_ = classes
        self._probabilities = probabilities

    def predict_proba(self, features: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
        assert features
        return [self._probabilities]


def _open_palm_landmarks() -> list[list[float]]:
    landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
    landmarks[0] = [0.0, 0.0, 0.0]
    landmarks[1] = [0.12, -0.08, 0.0]
    landmarks[2] = [0.22, -0.16, 0.0]
    landmarks[3] = [0.32, -0.24, 0.0]
    landmarks[4] = [0.42, -0.32, 0.0]
    for offset, indices in enumerate(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        base_x = -0.18 + offset * 0.12
        for joint_offset, index in enumerate(indices):
            landmarks[index] = [base_x, -0.2 - joint_offset * 0.18, 0.0]
    return landmarks
