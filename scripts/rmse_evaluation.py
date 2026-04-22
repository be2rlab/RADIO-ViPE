import argparse
import os
import numpy as np
from evo.core.trajectory import PoseTrajectory3D
import evo.core.sync as sync
from evo.tools import file_interface
from evo.core import metrics
from evo.tools import plot
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import csv


def transform_to_evo_trajectory(data):
    poses = []
    timestamps = []
    for i, pose in enumerate(data):
        if len(pose) >= 16:
            matrix_vals = list(map(float, pose[:16]))
            pose_matrix = np.array(matrix_vals).reshape(4, 4)

            # Skip poses with NaN/Inf (e.g. "tracking lost" frames in GT)
            if not np.all(np.isfinite(pose_matrix)):
                continue

            tx, ty, tz = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]
            quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]

            poses.append([tx, ty, tz, quat[3], quat[0], quat[1], quat[2]])
            timestamps.append(i * 0.1)

    poses = np.array(poses)
    return PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, 3:],
        timestamps=np.array(timestamps),
    )

    poses = np.array(poses)
    return PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, 3:],  # [qw, qx, qy, qz]
        timestamps=np.array(timestamps)
    )


def load_gt_poses_replica(gt_file):
    poses = []
    with open(gt_file, 'r') as f:
        for line in f:
            poses.append(line.strip().split())
    
    return transform_to_evo_trajectory(poses)


def load_gt_poses_tum(gt_file,rgb_file):
    timestamps = []
    with open(rgb_file,'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            image_name = line.split(' ')[1].split('/')[1]
            frame_timestamp = float(image_name.split('.png')[0])
            timestamps.append(frame_timestamp)
    poses = []
    final_poses = []
    gt_timestamps = []
    with open(gt_file, 'r') as f:
        for line in f:
            if '#' not in line:
                timestamp, tx, ty, tz, qx, qy, qz, qw = line.split()
                rotation = R.from_quat([qx, qy, qz, qw])
                rotation_matrix = rotation.as_matrix()
                pose = np.eye(4)
                pose[:3, :3] = rotation_matrix
                pose[:3, 3] = [tx, ty, tz]
                poses.append(pose.flatten())
                gt_timestamps.append(float(timestamp))
    gt_timestamps = np.array(gt_timestamps)
    for ts in timestamps:
        idx = np.argmin(np.abs(gt_timestamps - ts))
        final_poses.append(poses[idx])


    return transform_to_evo_trajectory(final_poses)


def load_gt_poses(args):
    if args.dataset == "replica":
        gt_file = os.path.join(args.gt_folder, args.scene_name, "traj.txt")
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        return load_gt_poses_replica(gt_file)
    if args.dataset == "tum":
        rgb_file = os.path.join(args.gt_folder, args.scene_name, "rgb.txt")
        gt_file = os.path.join(args.gt_folder, args.scene_name, "groundtruth.txt")
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        if not os.path.exists(rgb_file):
            raise FileNotFoundError(f"rgb file not found: {rgb_file}")
        return load_gt_poses_tum(gt_file,rgb_file)
    else: 
        raise Exception(f"{args.dataset} is not supported!") 


def load_slam_poses(npz_file):
    data = np.load(npz_file)
    poses = []
    for pose in data['data']:
        poses.append(pose.flatten())

    return transform_to_evo_trajectory(poses)


def align_trajectories(gt_traj, slam_traj):
    max_diff = 0.01
    gt_traj, slam_traj = sync.associate_trajectories(gt_traj, slam_traj, max_diff)

    slam_traj.align(gt_traj)

    return gt_traj, slam_traj


def calculate_rmse(gt_traj, slam_traj):
    # Compute APE using evo
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data((gt_traj, slam_traj))
    
    rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)

    return rmse


def draw_plot(traj_est, traj_ref, save_path, scene_name):
    os.makedirs(save_path, exist_ok=True)
    fig = plt.figure()
    traj_by_label = {"SLAM": traj_est, "GT": traj_ref}
    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    fig.savefig(f"{save_path}/{scene_name}.png")


def write_to_csv(csv_file, scene_name, rmse):
    data = {}
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    data[row[0]] = row[1]
    
    data[scene_name] = str(rmse)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for scene, value in data.items():
            writer.writerow([scene, value])


def main():
    parser = argparse.ArgumentParser(description='Compute RMSE ATE error')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (replica, TUM, etc.)")
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to ground truth folder')
    parser.add_argument('--results_folder', type=str, required=True, help='Path to results folder')
    parser.add_argument('--scene_name', type=str, required=True, help='Name of the scene')
    parser.add_argument('--metrics_path', type=str, required=True, help='Path to CSV file to save results')
    parser.add_argument('--plot', action='store_true', help='Generate plot visualization')
    parser.add_argument('--save_plot', type=str, help='Path to save plot image')
    
    args = parser.parse_args()
    
    slam_file = os.path.join(args.results_folder, "pose", f"{args.scene_name}.npz")
    
    if not os.path.exists(slam_file):
        raise FileNotFoundError(f"SLAM results file not found: {slam_file}")
    
    # print(f"Loading ground truth from: {gt_file}")
    gt_traj = load_gt_poses(args)
    
    # print(f"Loading SLAM poses from: {slam_file}")
    slam_traj = load_slam_poses(slam_file)
    
    # print(f"GT poses: {len(gt_traj.positions_xyz)}, SLAM poses: {len(slam_traj.positions_xyz)}")
        
    gt_traj, slam_traj = align_trajectories(gt_traj, slam_traj)
    rmse = calculate_rmse(gt_traj, slam_traj)
    if args.plot:
        draw_plot(slam_traj, gt_traj, args.save_plot, args.scene_name)

    write_to_csv(args.metrics_path, args.scene_name, rmse)
    
    print("RMSE: " + str(rmse))


if __name__ == "__main__":
    main()