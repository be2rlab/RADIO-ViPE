# Grounding

Highlight object instances in a SLAM map by text prompt. Uses RADSeg + SigLIP2 embeddings, with DBSCAN producing one bounding box per instance.

## Usage

Show the RGB cloud only:

```bash
python -m scripts.grounding /path/to/map.pt --pca-basis /path/to/pca.pt
```

Ground a prompt with red points + per-instance boxes:

```bash
python -m scripts.grounding /path/to/map.pt \
  --pca-basis /path/to/pca.pt \
  --ground "chair" \
  --show-object-points
```

Add `--no-bbox` for points only — useful when instances merge into one giant box.

## Parameter tuning

### Grounding

| Flag | Default | When to change |
|---|---|---|
| `--ground` | — | Text prompt. If multiple are passed with `;`, only the first is used. |
| `--threshold` | `0.25` | Starting cosine-similarity cutoff. The script lowers it adaptively, so treat this as an upper bound. **Lower** (e.g. `0.15`) if nothing matches; check the printed `min/max/mean/std` to calibrate. |

### DBSCAN — most scene-dependent

| Flag | Default | When to change |
|---|---|---|
| `--cluster-eps` | `0.1` m | Neighborhood radius. Too small → one object splits across boxes; too large → multiple instances merge. |
| `--cluster-min-samples` | `10` | Core-point threshold. Raise to reject more noise. |

### Visualization

| Flag | Default | Effect |
|---|---|---|
| `--show-object-points` | off | Red highlight points on the RGB cloud. |
| `--no-bbox` | off | Skip bounding boxes. |
| `--bbox-expansion` | `1.0` | Multiplier on box size. |

## Quick troubleshooting

- **No points found** → lower `--threshold`.
- **One giant box** → lower `--cluster-eps` (or use `--no-bbox`).
- **One object split into many boxes** → raise `--cluster-eps`.
- **Tiny stray boxes** → raise `--cluster-min-samples` or `min_cluster_size`.