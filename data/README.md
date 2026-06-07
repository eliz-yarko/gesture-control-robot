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

## Independent control/holdout split workflow

Do not use augmentation or derived landmark windows as independent control/holdout samples.
When a compatible external subset is available as class-named folders, first build one source
manifest and then split it with group-safe stratification:

```powershell
python scripts/build_manifest.py `
  --input data/external/hagrid_v2_subset `
  --dataset hagrid_v2 `
  --output data/processed/benchmark_inputs/hagrid_v2_subset_manifest.csv `
  --condition external_static `
  --distance unknown `
  --limit-per-class 40 `
  --include-unknown

python scripts/split_manifest.py `
  --input data/processed/benchmark_inputs/hagrid_v2_subset_manifest.csv `
  --train-output data/processed/training/hagrid_v2_subset_train_manifest.csv `
  --control-output data/processed/benchmark_inputs/hagrid_v2_subset_control_manifest.csv `
  --holdout-output data/processed/benchmark_inputs/hagrid_v2_subset_holdout_manifest.csv `
  --control-count 150 `
  --holdout-count 70 `
  --seed 42
```

The splitter keeps each `sample_id` group in only one split. If the source subset is too small
for both requested targets, the shortfall is shared between control and holdout instead of
starving holdout. The current local own-control data has only 96 valid cached landmark samples,
so an external compatible subset is still required to reach the 120-150+ control and 56-70+
holdout targets without leaking train-derived examples into evaluation.

For the HaGRID 30k 384p sample zip, extract only compatible static classes first:

```powershell
python scripts/extract_hagrid_zip_subset.py `
  --zip data/external/hagrid-sample-30k-384p.zip `
  --output-root data/external/hagrid_static_subset/all `
  --class call `
  --class palm `
  --class fist `
  --class like `
  --class dislike `
  --class peace `
  --class three `
  --class ok `
  --limit-per-class 140 `
  --seed 42
```

The current extracted subset has 1120 raw images across eight mapped labels. After landmark
filtering, it provides 295 train, 144 control, and 78 holdout valid samples. The tuned v10
own+HaGRID static candidate reaches HaGRID holdout accuracy `0.885` and macro F1 `0.905`,
but it remains an ablation rather than a live replacement because it still causes one false
`THUMB_DOWN` command on the own dense-valid control set.

The safer runtime candidate keeps the stable v3 own-control static model as primary and uses
the tuned v10 static model only as a low-confidence fallback:

```powershell
python scripts/evaluate_manifest.py `
  --manifest data/processed/benchmark_inputs/own_control_test_landmarks_dense_valid_manifest.csv `
  --output data/benchmarks/own_control_static_v3_primary_v10_tuned_fallback_p040_dynamic_v3_wave038_pipeline_dense_valid_predictions.csv `
  --classifier-mode pipeline `
  --static-model models/static_gesture_classifier_windowed_v3_open_ipn_min5.joblib `
  --secondary-static-model models/static_gesture_classifier_windowed_v10_own_hagrid_static.joblib `
  --secondary-static-threshold-profile models/hagrid_static_v10_safety_threshold_profile.json `
  --primary-static-min-confidence 0.4 `
  --dynamic-model models/dynamic_gesture_classifier_windowed_v3_open_ipn_min5.joblib `
  --dynamic-threshold-profile models/dynamic_v3_wave_lr_safety_threshold_profile.json `
  --frame-stride 1 `
  --max-frames 120 `
  --frame-width 480 `
  --frame-height 640 `
  --mirror-frame
```

This ensemble preserves the own dense-valid result (`accuracy=0.952`, `macro-F1=0.972`,
false confirmed command rate `0.000`) while keeping the HaGRID static subset at the tuned
v10 level (`control accuracy=0.868`, `holdout accuracy=0.885`). The current own holdout has
only 31 valid landmark samples and reaches `accuracy=0.871`, so it is one correct prediction
below `0.875` and still below the requested 56-70+ independent holdout count.

## Landmark cache workflow

For iterative training, export landmarks once and train/evaluate on JSON sequences instead of
running MediaPipe on every experiment:

```powershell
python scripts/export_landmark_manifest.py `
  --manifest data/processed/benchmark_inputs/own_control_train_manifest.csv `
  --output-manifest data/processed/benchmark_inputs/own_control_train_landmarks_manifest.csv `
  --output-dir data/processed/benchmark_inputs/own_control_train_landmarks_dense `
  --frame-stride 1 `
  --max-frames 120 `
  --frame-width 480 `
  --frame-height 640 `
  --mirror-frame `
  --continue-on-error

python scripts/export_landmark_manifest.py `
  --manifest data/processed/benchmark_inputs/own_control_test_manifest.csv `
  --output-manifest data/processed/benchmark_inputs/own_control_test_landmarks_dense_manifest.csv `
  --output-dir data/processed/benchmark_inputs/own_control_test_landmarks_dense `
  --frame-stride 1 `
  --max-frames 120 `
  --frame-width 480 `
  --frame-height 640 `
  --mirror-frame `
  --continue-on-error
```

The dense cache keeps all own-control videos in the training/evaluation protocol. The older
`min5` filtered manifests are useful only as a diagnostic slice: they remove samples where
MediaPipe initially produced too few landmarks, but they should not be treated as the final
own-data result because they hide difficult local videos.

Train and evaluate the dense v3 landmark models:

```powershell
python scripts/train_gesture_models.py `
  --manifest data/processed/benchmark_inputs/own_control_train_landmarks_dense_manifest.csv `
  --static-output models/static_gesture_classifier_windowed_v3_dense.joblib `
  --dynamic-output models/dynamic_gesture_classifier_windowed_v3_dense.joblib `
  --frame-stride 1 `
  --max-frames 120 `
  --static-min-confidence 0.25 `
  --dynamic-min-confidence 0.25 `
  --min-dynamic-window-points 5 `
  --dynamic-unknown-ratio 1.2 `
  --dynamic-augmentation-copies 2

python scripts/evaluate_manifest.py `
  --manifest data/processed/benchmark_inputs/own_control_test_landmarks_dense_manifest.csv `
  --output data/processed/benchmark_inputs/own_control_windowed_v3_dense40_test_predictions.csv `
  --classifier-mode auto `
  --static-model models/static_gesture_classifier_windowed_v3_dense.joblib `
  --dynamic-model models/dynamic_gesture_classifier_windowed_v3_dense.joblib `
  --dynamic-min-points 5 `
  --max-frames 40 `
  --fallback-to-heuristics `
  --continue-on-error

python scripts/benchmark.py `
  --input data/processed/benchmark_inputs/own_control_windowed_v3_dense40_test_predictions.csv `
  --output data/benchmarks/own_control_windowed_v3_dense40_test_results.csv

python scripts/tune_thresholds.py `
  --input data/processed/benchmark_inputs/own_control_windowed_v3_dense40_test_predictions.csv `
  --output models/own_control_windowed_v3_dense40_threshold_profile.json `
  --default-threshold 0.25 `
  --max-critical-fpr 0 `
  --step 0.05
```

Current dense own-control benchmark with temporal v3 features uses all 24 test videos:
raw macro F1 is `0.700`, accuracy is `0.667`, unknown rate is `0.125`, and critical
false-positive rate is `0.087`. The previous `min5` slice reached macro F1 `0.875`
and accuracy `0.850`, but it excluded difficult own-control videos and should be reported
as an ablation, not as the only final own-data result.

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

Merge the dense own-control train cache with IPN and train the open-data dynamic model:

```powershell
python scripts/retrain_open_data.py `
  --extra-manifest data/processed/benchmark_inputs/own_control_train_landmarks_dense_manifest.csv `
  --extra-manifest data/processed/training/ipn_hand_train_landmarks_min5_manifest.csv `
  --output-manifest data/processed/training/combined_own_dense_ipn_landmarks_manifest.csv `
  --include-unknown `
  --skip-train

python scripts/train_gesture_models.py `
  --manifest data/processed/training/combined_own_dense_ipn_landmarks_manifest.csv `
  --static-output models/static_gesture_classifier_windowed_v3_dense_open_ipn.joblib `
  --dynamic-output models/dynamic_gesture_classifier_windowed_v3_dense_open_ipn.joblib `
  --frame-stride 1 `
  --max-frames 120 `
  --static-min-confidence 0.25 `
  --dynamic-min-confidence 0.25 `
  --min-dynamic-window-points 5 `
  --dynamic-unknown-ratio 1.2 `
  --dynamic-augmentation-copies 2
```

This open-data model is kept as a benchmark/fine-tuning candidate, not as the
live default. The live demo should use the stable own-control models
`static_gesture_classifier.joblib` and `dynamic_gesture_classifier.joblib`
unless an experiment explicitly passes different model paths.

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
