# Running the ablation study

| Tier | Presets | Scenes | Runs | Backs |
|---|---|---|---|---|
| `core` | E9, K1, K3, K4, K8, E5 | 4 TUM walking | **24** | Table 4 — the kernel ladder |
| `consumers` | E0, E6, E7, E8 | 4 TUM walking | **16** | Table 5 — leave-one-out |
| `control` | E0, E5 | 4 TUM sitting | 8 | Suppl. Table 7 — low-dynamic check |
| `sens` | S_ts_065, S_ts_085, S_tm_025, S_tm_045 | 2 TUM walking | 8 | Suppl. Table 8 — thresholds |

**Required: 40 runs** (`core` + `consumers`).

## 1. Edit the paths

In `scripts/run_ablations.sh`, set for your server:

- `ROOT_DIR` — repository root (contains `run.py` and `scripts/`).
- `TUM_GT` — TUM dataset root; each scene is read from `$TUM_GT/<scene>/rgb`.
- `RESULTS_FOLDER` — where per-preset output goes.

`REPLICA_GT`, `REPLICA_SEM_GT` and `REPLICA_FIRST_POSE` are only consulted by
`--dataset replica`, which no tier uses.


## 2. Run the deferred unit tests

    python3 -m pytest tests/test_kernel_modes.py -v

These cover exactly the branching the core tier ablates: `use_variance`
(E5 vs K8), `stability_reduce`, `fixed` mode (K1/K3) and `none` mode (E9).


## 4. Smoke-test one preset

    scripts/run_ablations.sh --presets E5 \
        --scenes rgbd_dataset_freiburg3_walking_xyz --gpu 0

`E5` is the full system, so its ATE should reproduce the `QUORUM` row of the main TUM table (1.55 cm on `fr3/w/xyz`).

## 5. Run the sweep

    scripts/run_ablations.sh --tier core      --gpu 0    # 24 runs
    scripts/run_ablations.sh --tier consumers --gpu 0    # 16 runs

Optional:

    scripts/run_ablations.sh --tier control,sens --gpu 0  # 16 runs


## 6. Generate the tables

    python3 scripts/collect_ablations.py \
        --results ablation_results --tables paper/tables
    cd paper && latexmk -pdf main.tex

