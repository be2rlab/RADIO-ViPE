#!/bin/bash

export ROOT_DIR=/home/user/km-vipe
export GT_FOLDER=/data/tum
export RESULTS_FOLDER=$ROOT_DIR/tum_results_baseline
export CUDA_VISIBLE_DEVICES=0
export SCENE_NAMES=(
    # rgbd_dataset_freiburg1_360
    # rgbd_dataset_freiburg1_desk
    # rgbd_dataset_freiburg1_desk2
    # rgbd_dataset_freiburg1_floor
    # rgbd_dataset_freiburg1_plant
    # rgbd_dataset_freiburg1_room
    # rgbd_dataset_freiburg1_rpy
    # rgbd_dataset_freiburg1_teddy
    # rgbd_dataset_freiburg1_xyz
    rgbd_dataset_freiburg3_walking_xyz
    rgbd_dataset_freiburg3_walking_rpy
    rgbd_dataset_freiburg3_walking_halfsphere
    rgbd_dataset_freiburg3_walking_static
    rgbd_dataset_freiburg3_sitting_xyz
    rgbd_dataset_freiburg3_sitting_rpy
    rgbd_dataset_freiburg3_sitting_halfsphere
    rgbd_dataset_freiburg3_sitting_static
)


for SCENE_NAME in ${SCENE_NAMES[*]}
do
    printf "Running scene:   %s\n" "$SCENE_NAME"
    mkdir -p $RESULTS_FOLDER/profiling

    CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python3 $ROOT_DIR/run.py \
        pipeline=tum \
        streams=frame_dir_stream \
        streams.base_path=$GT_FOLDER/$SCENE_NAME/rgb \
        streams.scene_name=$SCENE_NAME \
        pipeline.output.save_artifacts=true \
        pipeline.output.path=$RESULTS_FOLDER \
        profiler.output=$RESULTS_FOLDER/profiling/${SCENE_NAME}.txt
        # pipeline.slam.dataset.sequence_name=$SCENE_NAME \
        # pipeline.slam.keyframe_depth=dataset \

    python $ROOT_DIR/scripts/rmse_evaluation.py \
        --dataset "tum" \
        --gt_folder "$GT_FOLDER" \
        --results_folder "$RESULTS_FOLDER" \
        --scene_name "$SCENE_NAME" \
        --metrics_path "$RESULTS_FOLDER/metrics.csv" \
        --plot \
        --save_plot "$RESULTS_FOLDER/rmse" \

done 