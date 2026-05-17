# SLAM Map Grounding Visualizer

An interactive 3D visualizer for SLAM maps with **open-vocabulary text-based object grounding**. Renders a dense RGB point cloud in [Rerun](https://www.rerun.io/) and, given a text prompt, highlights matching object instances using language–vision embeddings (RADSeg + SigLIP2), clustered into per-instance oriented bounding boxes.

---

## Installation

Dependencies:

- `numpy`, `torch`, `scipy`, `scikit-learn`
- `rerun-sdk`
- The `vipe` package (provides `RADSegEncoder`)

```bash
pip install numpy torch scipy scikit-learn rerun-sdk
# plus your local install of vipe
```

---

## Quick start

Just show the RGB cloud:

```bash
python visualize_slam_map.py /path/to/map.pt --pca-basis /path/to/pca.pt
```

Ground a prompt and visualize matching points + per-instance bounding boxes:

```bash
python visualize_slam_map.py /path/to/map.pt \
  --pca-basis /path/to/pca.pt \
  --ground "chair" \
  --show-object-points
```

Points only, no boxes (better for dense multi-instance scenes):

```bash
python visualize_slam_map.py /path/to/map.pt \
  --pca-basis /path/to/pca.pt \
  --ground "chair" \
  --show-object-points \
  --no-bbox
```

---

## How it works

The pipeline runs five stages:

1. **Load.** The map (`.pt`) is loaded onto CPU or GPU. It must contain `dense_disp_xyz`, `dense_disp_rgb`, and either pre-decompressed (`dense_disp_embeddings_full`) or PCA-compressed (`dense_disp_embeddings`) per-point language embeddings.
2. **Decompress.** If only compressed embeddings exist and `--pca-basis` is supplied, `PcaCompressor` projects them back to the full embedding space.
3. **Align.** If `--gt-poses-path` is provided, the first 4×4 pose is applied to the cloud. The scene's principal axes are then computed via PCA on the cloud — these define a single orientation that all bounding boxes share.
4. **Ground.** The text prompt is encoded with RADSeg/SigLIP2. Cosine similarity is computed against every point embedding. An **adaptive threshold** walks the threshold down from `--threshold` until either the selected points form a coherent cluster (std above a target) or a minimum threshold is reached (top-1% fallback).
5. **Cluster & box.** Selected points are clustered with **DBSCAN**. Clusters smaller than `min_cluster_size` are dropped. Each remaining cluster gets its own oriented bounding box, sized using the 7th/93rd percentiles of its extent (robust to outliers).

---

## Tuning guide

### Grounding parameters

| Flag | Default | What to change |
|---|---|---|
| `--ground` | none | The text prompt. If you pass several with `;`, **only the first is used**. |
| `--threshold` | `0.25` | Starting cosine-similarity threshold. **Lower** (e.g. `0.15`) if no points are found for a prompt that should match. **Raise** if too much spurious activation. The adaptive logic only lowers, never raises, so treat this as the upper bound. |

The script prints similarity stats (`min/max/mean/std`) every run — use those to calibrate `--threshold`. If `max` is around `0.20`, a default of `0.25` will never trigger and you'll fall through to the top-1% fallback.

**Internal grounding constants** (hardcoded in `find_grounding_threshold`, edit in source if needed):

- `target_std = 0.015` — minimum spread the selected points must have to count as a real cluster (SigLIP2-calibrated).
- `min_threshold = 0.05` — noise floor below which the adaptive search gives up.
- `min_points = 10` — minimum number of selected points before the std check kicks in.

These are stable across most scenes; only touch them if you swap the language model.

### DBSCAN parameters — **the main thing to tune per scene**

DBSCAN turns the cloud of "matched" points into individual instances. **The right values depend heavily on the spatial scale of your scene.** This is by far the most scene-dependent part of the pipeline.

| Flag | Default | When to change |
|---|---|---|
| `--cluster-eps` | `0.1` m | Neighborhood radius. **The single most important parameter.** Set it to roughly the gap between adjacent instances of your object. Too small → one real object splits into multiple clusters. Too large → multiple instances merge into one. |
| `--cluster-min-samples` | `10` | DBSCAN's core-point threshold. Raise to reject more noise; lower if genuinely sparse instances are being lost as noise. |

Rule-of-thumb `eps` by scene type:

| Scene | Suggested `--cluster-eps` |
|---|---|
| Tabletop / small objects | `0.03 – 0.08` |
| Indoor furniture (chairs, lamps) | `0.15 – 0.30` |
| Rooms / large indoor objects | `0.30 – 0.50` |
| Outdoor (cars, signs) | `0.5 – 1.0` |

There is **also** a `min_cluster_size = 60` parameter inside `cluster_object_points` (in source, not exposed as a CLI flag) that discards clusters smaller than 60 points after DBSCAN runs. **Raise** this if tiny false-positive blobs are slipping through; **lower** if real small instances are being filtered out. To expose it as a CLI flag, add a `--min-cluster-size` argument and forward it through `cluster_object_points(...)`.

### Visualization flags

| Flag | Default | Effect |
|---|---|---|
| `--show-object-points` | off | Render grounded points in **red** on top of the RGB cloud. |
| `--no-bbox` | off | Skip bounding boxes (useful for messy multi-instance scenes). |
| `--bbox-expansion` | `1.0` | Multiplicative size factor for each box. `1.05` = 5 % larger. |

### RADSeg parameters

These mirror the upstream RADSeg config and rarely need changing unless you swap the backbone:

- `--model-version` (default `c-radio_v3-b`)
- `--lang-model` (default `siglip2`)
- `--scra-scaling`, `--scga-scaling` (default `10.0`)
- `--window-size`, `--window-stride` (default `336`, `224`)

---

## Reading the Rerun viewer

Entity tree:

| Entity | Contents |
|---|---|
| `world/points/rgb` | The original SLAM cloud (colored). |
| `world/objects/{prompt}_points` | Red highlight points (when `--show-object-points`). |
| `world/objects/{prompt}/instance_i` | One oriented green bounding box per detected instance. **Sorted by size — `instance_0` is the largest.** |
| `world/axes` | World-origin coordinate frame. |

Each entity can be toggled on/off independently from Rerun's blueprint panel — handy for comparing raw cloud vs. highlight vs. boxes.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No points found for '<prompt>'` | `--threshold` too high, or prompt doesn't match anything. | Inspect printed `min/max/mean/std` of similarity. Lower `--threshold` toward `mean + 1–2·std`. |
| `Falling back to top-K points` | No threshold produced a coherent cluster. | Either the prompt is a poor match or the similarity distribution is too flat. The fallback still renders something — check whether it makes sense before trusting it. |
| One giant bounding box covering most of the map | `--cluster-eps` too large, instances merged. | Lower `--cluster-eps`. If even small `eps` merges them, the points themselves are too scattered — try `--no-bbox` and view the red highlight instead. |
| Several boxes splitting one real object | `--cluster-eps` too small. | Raise it. |
| Tiny extraneous boxes scattered around | False-positive blobs surviving DBSCAN. | Raise `min_cluster_size` (in source) or `--cluster-min-samples`. |
| Box is rotated weirdly | The scene's PCA-derived orientation is unusual (e.g. a long thin scene with one axis dominating). | Acceptable side effect: all boxes share one orientation. If you want truly per-instance orientations, compute PCA on each cluster instead of on the full cloud. |

---

## File reference

- `visualize_slam_map.py` — the script.
- The map `.pt` is expected to contain at minimum:
  - `dense_disp_xyz`: `(N, 3)` float tensor — point positions.
  - `dense_disp_rgb`: `(N, 3)` float tensor in `[0, 1]` — point colors.
  - `dense_disp_embeddings` *or* `dense_disp_embeddings_full`: `(N, D)` per-point language embeddings.
  - `dense_disp_embedding_valid` *(optional)*: `(N,)` bool mask of valid embeddings.