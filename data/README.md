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
