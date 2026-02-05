#!/bin/bash

export ROOT_DIR=/home/user/km-vipe
export GT_FOLDER=/data/aria_Replica/
export RESULTS_FOLDER=$ROOT_DIR/aria_results
export DATASET_NAME="aria"
export SCENE_NAMES=(
Apartment_release_clean_seq134_M1292    
Apartment_release_multiuser_clean_seq112_M1292        
# Apartment_release_multiuser_party_seq140_M1292
# Apartment_release_multiskeleton_party_seq101_M1292
# Apartment_release_multiskeleton_party_seq102_M1292
# Apartment_release_meal_skeleton_seq133_M1292
# Apartment_release_decoration_skeleton_seq131_M1292
)

for SCENE_NAME in ${SCENE_NAMES[*]}
do
    printf "Running scene:   %s\n" "$SCENE_NAME"

    echo "Cleaning GPU memory..."
    sleep 3
    sudo fuser -v /dev/nvidia* 2>/dev/null | grep python && echo "Killing GPU python processes..." && sudo fuser -k /dev/nvidia* || echo "No active GPU processes."

    CUDA_LAUNCH_BLOCKING=1 python3 $ROOT_DIR/run.py \
        pipeline=aria \
        streams=frame_dir_stream \
        streams.base_path=$GT_FOLDER/$SCENE_NAME/results \
        streams.scene_name=$SCENE_NAME \
        streams.dataset_name=$DATASET_NAME \
        pipeline.output.save_artifacts=true \
        pipeline.output.path=$RESULTS_FOLDER \
        # pipeline.slam.dataset.sequence_name=$SCENE_NAME \
        # pipeline.slam.keyframe_depth=dataset \

    python $ROOT_DIR/scripts/rmse_evaluation.py \
        --dataset "replica" \
        --gt_folder "$GT_FOLDER" \
        --results_folder "$RESULTS_FOLDER" \
        --scene_name "$SCENE_NAME" \
        --metrics_path "$RESULTS_FOLDER/metrics.csv" \
        --plot \
        --save_plot "$RESULTS_FOLDER/rmse" \

done 