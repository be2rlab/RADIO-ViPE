#!/bin/bash
set -euo pipefail

# ---- EDIT THESE FOR YOUR SERVER --------------------------------------------
ROOT_DIR=${ROOT_DIR:-/home/user/km-vipe}
TUM_GT=${TUM_GT:-/data/tum}
REPLICA_GT=${REPLICA_GT:-/data/Replica}
RESULTS_FOLDER=${RESULTS_FOLDER:-$ROOT_DIR/ablation_results}
REPLICA_SEM_GT=${REPLICA_SEM_GT:-/data/NiceReplicaDataset/}
REPLICA_FIRST_POSE=${REPLICA_FIRST_POSE:-/data/Replica}
# ----------------------------------------------------------------------------

DATASET=tum
GPU=0
PRESETS=""
SCENES=""
TIER=""
DRY_RUN=false

TUM_WALKING="rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_rpy,\
rgbd_dataset_freiburg3_walking_halfsphere,rgbd_dataset_freiburg3_walking_static"
TUM_SITTING="rgbd_dataset_freiburg3_sitting_xyz,rgbd_dataset_freiburg3_sitting_rpy,\
rgbd_dataset_freiburg3_sitting_halfsphere,rgbd_dataset_freiburg3_sitting_static"
TUM_ALL="$TUM_WALKING,$TUM_SITTING"
REPLICA_ALL="room0,office0,office2"

# Two walking scenes with the largest ablation spread, used by the optional
# sensitivity tier to keep it to 8 runs.
TUM_SENS="rgbd_dataset_freiburg3_walking_xyz,rgbd_dataset_freiburg3_walking_halfsphere"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --presets) PRESETS="$2"; shift 2 ;;
        --scenes)  SCENES="$2";  shift 2 ;;
        --tier)    TIER="$2";    shift 2 ;;
        --gpu)     GPU="$2";     shift 2 ;;
        --results) RESULTS_FOLDER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

JOBS=()
add_tier() {
    case "$1" in
        core)      JOBS+=("E9,K1,K3,K4,K8,E5|$TUM_WALKING|tum") ;;
        consumers) JOBS+=("E0,E6,E7,E8|$TUM_WALKING|tum") ;;
        control)   JOBS+=("E0,E5|$TUM_SITTING|tum") ;;
        sens)      JOBS+=("S_ts_065,S_ts_085,S_tm_025,S_tm_045|$TUM_SENS|tum") ;;
        all)       add_tier core; add_tier consumers; add_tier control; add_tier sens ;;
        *) echo "Unknown tier: $1 (expected core|consumers|control|sens|all)" >&2; exit 2 ;;
    esac
}

if [[ -n "$TIER" ]]; then
    if [[ -n "$PRESETS" || -n "$SCENES" ]]; then
        echo "--tier cannot be combined with --presets/--scenes" >&2
        exit 2
    fi
    IFS=',' read -ra TIER_LIST <<< "$TIER"
    for T in "${TIER_LIST[@]}"; do add_tier "$T"; done
else
    if [[ -z "$PRESETS" ]]; then
        echo "Either --tier (core|consumers|control|sens|all) or --presets is required." >&2
        exit 2
    fi
    if [[ -z "$SCENES" ]]; then
        [[ "$DATASET" == "tum" ]] && SCENES=$TUM_ALL || SCENES=$REPLICA_ALL
    fi
    JOBS+=("$PRESETS|$SCENES|$DATASET")
fi

FAILURES=()
TOTAL=0

run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        printf '        +'; printf ' %q' "$@"; printf '\n'
        return 0
    fi
    "$@"
}

for JOB in "${JOBS[@]}"; do
    IFS='|' read -r JOB_PRESETS JOB_SCENES JOB_DATASET <<< "$JOB"

    if [[ "$JOB_DATASET" == "tum" ]]; then
        GT_FOLDER=$TUM_GT
    else
        GT_FOLDER=$REPLICA_GT
    fi

    IFS=',' read -ra PRESET_LIST <<< "$JOB_PRESETS"
    IFS=',' read -ra SCENE_LIST  <<< "$JOB_SCENES"

    echo "=============================================================="
    echo "dataset=$JOB_DATASET gpu=$GPU results=$RESULTS_FOLDER"
    echo "presets: ${PRESET_LIST[*]}"
    echo "scenes:  ${#SCENE_LIST[@]} sequence(s)"
    echo "runs:    $(( ${#PRESET_LIST[@]} * ${#SCENE_LIST[@]} ))"
    echo "=============================================================="

    for PRESET in "${PRESET_LIST[@]}"; do
        # Resolved up front so an unknown preset name fails immediately, before
        # any GPU time is spent, rather than midway through the sweep.
        OVERRIDES=$(python3 "$ROOT_DIR/scripts/ablation_presets.py" "$PRESET")
        OUT="$RESULTS_FOLDER/$PRESET"
        [[ "$DRY_RUN" == true ]] || mkdir -p "$OUT/profiling"

        for SCENE in "${SCENE_LIST[@]}"; do
            SLAM_SKIPPED=false
            TOTAL=$(( TOTAL + 1 ))

            # A finished run leaves its trajectory at pose/<scene>.npz
            # (see vipe/utils/io.py:pose_path and rmse_evaluation.py's slam_file).
            if [[ -f "$OUT/pose/${SCENE}.npz" ]]; then
                echo "  [skip] $PRESET / $SCENE (already done)"
                SLAM_SKIPPED=true
            else
                echo "  [run ] $PRESET / $SCENE"

                if ! run_cmd env CUDA_VISIBLE_DEVICES=$GPU python3 "$ROOT_DIR/run.py" \
                    pipeline="$JOB_DATASET" \
                    streams=frame_dir_stream \
                    streams.base_path="$GT_FOLDER/$SCENE/rgb" \
                    streams.scene_name="$SCENE" \
                    pipeline.output.save_artifacts=true \
                    pipeline.output.save_viz=false \
                    pipeline.output.path="$OUT" \
                    pipeline.slam.visualize=false \
                    pipeline.slam.sequence_name="$SCENE" \
                    pipeline.slam.pca_state_path="$OUT/vipe/${SCENE}_pca_basis.pt" \
                    profiler.output="$OUT/profiling/${SCENE}.txt" \
                    $OVERRIDES; then
                    echo "  [FAIL] $PRESET / $SCENE (run.py)" >&2
                    FAILURES+=("$PRESET/$SCENE (run.py)")
                    continue
                fi

                if ! run_cmd python3 "$ROOT_DIR/scripts/rmse_evaluation.py" \
                    --dataset "$JOB_DATASET" \
                    --gt_folder "$GT_FOLDER" \
                    --results_folder "$OUT" \
                    --scene_name "$SCENE" \
                    --metrics_path "$OUT/metrics.csv"; then
                    echo "  [FAIL] $PRESET / $SCENE (rmse_evaluation.py)" >&2
                    FAILURES+=("$PRESET/$SCENE (rmse_evaluation.py)")
                    continue
                fi
            fi

            if [[ "$JOB_DATASET" == "replica" ]]; then
                SEM_OUT="$RESULTS_FOLDER/eval_out/$PRESET"


                if [[ "$SLAM_SKIPPED" == true ]] \
                    && grep -qs "^${SCENE}," "$SEM_OUT"/*/*_global_summary_results.csv 2>/dev/null; then
                    echo "  [skip-sem] $PRESET / $SCENE (semantic eval already done)"
                else
                    echo "  [run-sem] $PRESET / $SCENE"

                    if ! run_cmd python3 "$ROOT_DIR/scripts/semseg_eval.py" \
                        --config-name semseg_configs/replica_kmvipe \
                        eval_out="$SEM_OUT" \
                        pred_dir="$OUT/vipe/" \
                        dataset.scene_name="$SCENE" \
                        gt_dir="$REPLICA_SEM_GT" \
                        first_pose_dir="$REPLICA_FIRST_POSE"; then
                        echo "  [FAIL-SEM] $PRESET/$SCENE" >&2
                        FAILURES+=("$PRESET/$SCENE (semseg_eval.py)")
                        continue
                    fi
                fi
            fi
        done
    done
done

echo ""
echo "Finished $TOTAL job(s). Collect with:"
echo "  python3 scripts/collect_ablations.py --results $RESULTS_FOLDER \\"
echo "      --tables ../paper/tables"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "" >&2
    echo "${#FAILURES[@]} job(s) failed:" >&2
    for F in "${FAILURES[@]}"; do
        echo "  - $F" >&2
    done
    exit 1
fi
