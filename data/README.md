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

Supported manifest presets:

- `own_control` for locally recorded class folders named after project gestures.
- `hagrid` / `hagrid_v2` for static gesture subsets.
- `jester` for selected dynamic gesture folders.
- `ipn_hand` for selected IPN Hand classes such as `G05`, `G06`, and `G10`.
