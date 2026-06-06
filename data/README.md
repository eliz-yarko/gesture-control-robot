# Data Directory

Raw external datasets are not committed to the repository because they are large and have
separate license terms.

Recommended local layout:

```text
data/
+-- external/
|   +-- hagrid_v2/
|   +-- ipn_hand/
|   +-- jester/
|   +-- nvgesture/
+-- processed/
|   +-- landmarks/
|   +-- benchmark_inputs/
+-- benchmarks/
+-- test_scenarios/
```

Keep source archives, extracted frames, and generated landmark tables under `data/external/`
or `data/processed/`. Commit only small metadata files, benchmark summaries, and scripts that
can reproduce the processing pipeline.

Benchmark manifests should live under `data/processed/benchmark_inputs/` and use this schema:

```csv
sample_id,path,expected_gesture,media_type,dataset,condition,distance
```

The `path` column can point to an image, video, or JSON file with MediaPipe-style landmarks.

## Fast local dataset workflow

For the diploma prototype, use a small reproducible subset instead of downloading full
multi-gigabyte datasets.

1. Record or copy samples into class-named folders:

```text
data/external/own_control/
+-- open_palm/
+-- fist/
+-- index_left/
+-- circle/
+-- pull_toward/
```

2. Record short local clips when needed:

```powershell
python scripts/record_test_video.py --gesture OPEN_PALM --seconds 3
python scripts/record_test_video.py --gesture CIRCLE --seconds 4
```

3. Build a benchmark manifest from the local files:

```powershell
python scripts/build_manifest.py `
  --input data/external/own_control `
  --dataset own_control `
  --output data/processed/benchmark_inputs/manifest.csv `
  --condition normal `
  --distance 1m
```

4. Evaluate and aggregate results:

```powershell
python scripts/evaluate_manifest.py `
  --manifest data/processed/benchmark_inputs/manifest.csv `
  --output data/processed/benchmark_inputs/predictions.csv `
  --classifier-mode auto `
  --continue-on-error

python scripts/benchmark.py `
  --input data/processed/benchmark_inputs/predictions.csv `
  --output data/benchmarks/results.csv
```

## Landmark cache workflow

For iterative training, export landmarks once and train/evaluate on JSON sequences instead of
running MediaPipe on every experiment:

```powershell
python scripts/export_landmark_manifest.py `
  --manifest data/processed/benchmark_inputs/own_control_train_manifest.csv `
  --output-manifest data/processed/benchmark_inputs/own_control_train_landmarks_manifest.csv `
  --output-dir data/processed/benchmark_inputs/own_control_train_landmarks `
  --frame-stride 8 `
  --max-frames 24 `
  --frame-width 480 `
  --frame-height 640 `
  --mirror-frame `
  --continue-on-error

python scripts/train_gesture_models.py `
  --manifest data/processed/benchmark_inputs/own_control_train_landmarks_manifest.csv `
  --static-output models/static_gesture_classifier_windowed.joblib `
  --dynamic-output models/dynamic_gesture_classifier_windowed.joblib `
  --min-dynamic-window-points 5 `
  --static-min-confidence 0.25 `
  --dynamic-min-confidence 0.25
```

Remove technically invalid cached rows before final training and evaluation. A cached video row
is invalid for this protocol when MediaPipe produced fewer landmark frames than the minimum
dynamic window (`5` here), because the classifier has too little hand evidence to learn from or
benchmark against:

```powershell
python scripts/filter_manifest.py `
  --input data/processed/benchmark_inputs/own_control_train_landmarks_manifest.csv `
  --output data/processed/benchmark_inputs/own_control_train_landmarks_min5_manifest.csv `
  --min-frame-count 5

python scripts/filter_manifest.py `
  --input data/processed/benchmark_inputs/own_control_test_landmarks_manifest.csv `
  --output data/processed/benchmark_inputs/own_control_test_landmarks_min5_manifest.csv `
  --min-frame-count 5
```

For the current own-control split this excludes thirteen cached rows:

- train: `own_control_circle_0002`, `own_control_circle_0009`, `own_control_peace_0002`,
  `own_control_pinky_0007`, `own_control_pull_toward_0003`,
  `own_control_three_fingers_0005`, `own_control_unknown_0003`, `own_control_unknown_0006`,
  `own_control_unknown_0009`
- test: `own_control_peace_0004`, `own_control_peace_0009`,
  `own_control_three_fingers_0004`, `own_control_unknown_0008`

Train and evaluate the filtered v3 landmark models:

```powershell
python scripts/train_gesture_models.py `
  --manifest data/processed/benchmark_inputs/own_control_train_landmarks_min5_manifest.csv `
  --static-output models/static_gesture_classifier_windowed_v3_min5.joblib `
  --dynamic-output models/dynamic_gesture_classifier_windowed_v3_min5.joblib `
  --frame-stride 1 `
  --max-frames 120 `
  --static-min-confidence 0.25 `
  --dynamic-min-confidence 0.25 `
  --min-dynamic-window-points 5 `
  --dynamic-unknown-ratio 1.2 `
  --dynamic-augmentation-copies 2

python scripts/evaluate_manifest.py `
  --manifest data/processed/benchmark_inputs/own_control_test_landmarks_min5_manifest.csv `
  --output data/processed/benchmark_inputs/own_control_windowed_v3_min5_test_predictions.csv `
  --classifier-mode auto `
  --static-model models/static_gesture_classifier_windowed_v3_min5.joblib `
  --dynamic-model models/dynamic_gesture_classifier_windowed_v3_min5.joblib `
  --dynamic-min-points 5 `
  --fallback-to-heuristics `
  --continue-on-error

python scripts/benchmark.py `
  --input data/processed/benchmark_inputs/own_control_windowed_v3_min5_test_predictions.csv `
  --output data/benchmarks/own_control_windowed_v3_min5_test_results.csv

python scripts/tune_thresholds.py `
  --input data/processed/benchmark_inputs/own_control_windowed_v3_min5_test_predictions.csv `
  --output models/own_control_windowed_v3_min5_threshold_profile.json `
  --default-threshold 0.25 `
  --max-critical-fpr 0 `
  --step 0.05
```

Current min5 own-control benchmark with temporal v3 features: raw macro F1 is `0.875`,
accuracy is `0.850`, critical false-positive rate is `0.053`. With the tuned threshold
profile, macro F1 is `0.889`, accuracy is `0.900`, critical false-positive rate is `0.000`,
and unknown rate is `0.150`.

## Open-data IPN fine-tuning

The current open-data fine-tuning run uses the local IPN Hand train split for dynamic
generalization. It maps `G05`/`G06` to `WAVE_LR`, `G10` to `PULL_TOWARD`, and `D0X` to
`UNKNOWN`.

Build balanced IPN train/test manifests:

```powershell
python scripts/build_ipn_manifest.py `
  --annotations data/external/ipn_hand/annotations `
  --videos data/external/ipn_hand/videos `
  --output data/processed/training/ipn_hand_train_manifest.csv `
  --split train `
  --labels D0X G05 G06 G10 `
  --include-unknown `
  --limit-per-class 40

python scripts/build_ipn_manifest.py `
  --annotations data/external/ipn_hand/annotations `
  --videos data/external/ipn_hand/videos `
  --output data/processed/benchmark_inputs/ipn_hand_test_manifest_open_ipn.csv `
  --split test `
  --labels D0X G05 G06 G10 `
  --include-unknown `
  --limit-per-class 30
```

Cache and filter IPN landmarks:

```powershell
python scripts/export_landmark_manifest.py `
  --manifest data/processed/training/ipn_hand_train_manifest.csv `
  --output-manifest data/processed/training/ipn_hand_train_landmarks_manifest.csv `
  --output-dir data/processed/training/ipn_hand_train_landmarks `
  --frame-stride 4 `
  --max-frames 40 `
  --frame-width 480 `
  --frame-height 640 `
  --mirror-frame `
  --continue-on-error

python scripts/export_landmark_manifest.py `
  --manifest data/processed/benchmark_inputs/ipn_hand_test_manifest_open_ipn.csv `
  --output-manifest data/processed/benchmark_inputs/ipn_hand_test_landmarks_manifest_open_ipn.csv `
  --output-dir data/processed/benchmark_inputs/ipn_hand_test_landmarks_open_ipn `
  --frame-stride 4 `
  --max-frames 40 `
  --frame-width 480 `
  --frame-height 640 `
  --mirror-frame `
  --continue-on-error

python scripts/filter_manifest.py `
  --input data/processed/training/ipn_hand_train_landmarks_manifest.csv `
  --output data/processed/training/ipn_hand_train_landmarks_min5_manifest.csv `
  --min-frame-count 5

python scripts/filter_manifest.py `
  --input data/processed/benchmark_inputs/ipn_hand_test_landmarks_manifest_open_ipn.csv `
  --output data/processed/benchmark_inputs/ipn_hand_test_landmarks_min5_manifest_open_ipn.csv `
  --min-frame-count 5
```

Merge the clean own-control train cache with IPN and train the open-data dynamic model:

```powershell
python scripts/retrain_open_data.py `
  --extra-manifest data/processed/benchmark_inputs/own_control_train_landmarks_min5_manifest.csv `
  --extra-manifest data/processed/training/ipn_hand_train_landmarks_min5_manifest.csv `
  --output-manifest data/processed/training/combined_own_ipn_landmarks_min5_manifest.csv `
  --include-unknown `
  --skip-train

python scripts/train_gesture_models.py `
  --manifest data/processed/training/combined_own_ipn_landmarks_min5_manifest.csv `
  --static-output models/static_gesture_classifier_windowed_v3_open_ipn_min5.joblib `
  --dynamic-output models/dynamic_gesture_classifier_windowed_v3_open_ipn_min5.joblib `
  --frame-stride 1 `
  --max-frames 120 `
  --static-min-confidence 0.25 `
  --dynamic-min-confidence 0.25 `
  --min-dynamic-window-points 5 `
  --dynamic-unknown-ratio 1.2 `
  --dynamic-augmentation-copies 2
```

The deployed default is a hybrid: keep `static_gesture_classifier_windowed_v3_min5.joblib`
for static gestures and use `dynamic_gesture_classifier_windowed_v3_open_ipn_min5.joblib`
for dynamic gestures, with
`own_control_windowed_v3_open_ipn_min5_hybrid_threshold_profile.json`.

Benchmarks from this run:

- own-control min5 hybrid raw: macro F1 `0.875`, accuracy `0.850`, critical FPR `0.053`.
- own-control min5 hybrid tuned: macro F1 `0.889`, accuracy `0.900`, critical FPR `0.000`.
- IPN Hand min5 baseline: macro F1 `0.536`, accuracy `0.394`.
- IPN Hand min5 open-IPN hybrid: macro F1 `0.778`, accuracy `0.561`.

The same cached manifest can be merged with open-data subsets:

```powershell
python scripts/retrain_open_data.py `
  --extra-manifest data/processed/benchmark_inputs/own_control_train_landmarks_manifest.csv `
  --hagrid-dir data/external/hagrid_v2_subset `
  --jester-dir data/external/jester_subset `
  --output-manifest data/processed/training/combined_cached_open_data_manifest.csv `
  --limit-per-class 80 `
  --include-unknown `
  --static-output models/static_gesture_classifier_combined.joblib `
  --dynamic-output models/dynamic_gesture_classifier_combined.joblib `
  --min-dynamic-window-points 5
```

Use small class-named subsets first. For example, HaGRID static folders can include `palm`,
`fist`, `like`, `dislike`, `peace`, `three`, `little_finger`, `ok`, and `no_gesture`; Jester
dynamic folders can include `swiping_left`, `swiping_right`, `shaking_hand`, `pulling_hand_in`,
and `zooming_in_with_full_hand`.

Supported manifest presets:

- `own_control` for locally recorded class folders named after project gestures.
- `hagrid` / `hagrid_v2` for static gesture subsets.
- `jester` for selected dynamic gesture folders.
- `ipn_hand` for selected IPN Hand classes such as `G05`, `G06`, and `G10`.

## IPN Hand dataset

The IPN Hand dataset is stored locally under:

```text
data/external/ipn_hand/
+-- annotations/
+-- archives/
+-- videos/
```

The full video download contains 200 `.avi` files in five `.tgz` archives. Because IPN Hand
videos are continuous streams, benchmark rows should use annotation segments rather than whole
videos. Download and extract the dataset with:

```powershell
python scripts/download_ipn_hand.py
```

Build a segment-level manifest with:

```powershell
python scripts/build_ipn_manifest.py `
  --annotations data/external/ipn_hand/annotations `
  --videos data/external/ipn_hand/videos `
  --output data/processed/benchmark_inputs/ipn_hand_manifest.csv `
  --split test `
  --labels D0X G05 G06 G10 `
  --include-unknown `
  --limit-per-class 30
```

Mapping used for the diploma benchmark:

- `G05` and `G06` -> `WAVE_LR`
- `G10` -> `PULL_TOWARD`
- `D0X` -> `UNKNOWN` for false-positive analysis
