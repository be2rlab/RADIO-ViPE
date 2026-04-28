#!/bin/bash

export ROOT_DIR=/home/user/km-vipe
export GT_FOLDER=/data/scannet_v2/exported
export RESULTS_FOLDER=$ROOT_DIR/vipe_results_scannet
export CUDA_VISIBLE_DEVICES=0
export SCENE_NAMES=(
    scene0011_00
    scene0011_01
    scene0050_00
    scene0050_01
    scene0050_02
    scene0231_00
    scene0231_01
    scene0231_02
    scene0378_00
    scene0378_01
    scene0378_02
    scene0518_00
)

for SCENE_NAME in ${SCENE_NAMES[*]}
do
    printf "Running scene:   %s\n" "$SCENE_NAME"
    mkdir -p $RESULTS_FOLDER/profiling

    CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python3 $ROOT_DIR/run.py \
        pipeline=scannet \
        streams=frame_dir_stream \
        streams.base_path=$GT_FOLDER/$SCENE_NAME/color \
        streams.scene_name=$SCENE_NAME \
        pipeline.output.save_artifacts=true \
        pipeline.output.path=$RESULTS_FOLDER \
        pipeline.slam.sequence_name=$SCENE_NAME \
        pipeline.slam.pca_state_path=$RESULTS_FOLDER/vipe/${SCENE_NAME}_pca_basis.pt \
        profiler.output=$RESULTS_FOLDER/profiling/${SCENE_NAME}.txt


    python $ROOT_DIR/scripts/rmse_evaluation.py \
        --dataset "replica" \
        --gt_folder "$GT_FOLDER" \
        --results_folder "$RESULTS_FOLDER" \
        --scene_name "$SCENE_NAME" \
        --metrics_path "$RESULTS_FOLDER/metrics.csv" \
        --plot \
        --save_plot "$RESULTS_FOLDER/rmse" \

done 