#!/usr/bin/env python3
# experiments/run_duag_c.py
"""
Main entry point for DUAG-C experiments.
Usage: python experiments/run_duag_c.py --config experiments/configs/replica_multiagent_duag_c.yaml
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.types import GaussianMap, PoseGraph, RobotMessage
from core.pipeline.robot_node import RobotNode
from core.pipeline.local_slam_wrapper import LocalSLAMWrapper
from core.consensus.lie_algebra import se3_log


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class ReplicaMultiagentLoader:
    """
    Loads RGB-D frames and GT poses for one agent from the ReplicaMultiagent dataset.

    Structure:
        {scene}/{scene_lower}_part{agent+1}/results/frame{:06d}.jpg
        {scene}/{scene_lower}_part{agent+1}/results/depth{:06d}.png
        {scene}/{scene_lower}_part{agent+1}/traj.txt
    """

    def __init__(self, dataset_path: str, scene: str, agent_id: int, config: dict):
        self.dataset_path = Path(dataset_path)
        self.scene = scene
        self.agent_id = agent_id
        self.depth_scale = config["dataset"]["depth_scale"]
        self.frame_limit = config.get("frame_limit", -1)

        # Build part name: "Apart-0" -> "apart_0", agent 0 -> "apart_0_part1"
        scene_lower = scene.lower().replace("-", "_")
        part_name = f"{scene_lower}_part{agent_id + 1}"
        self.agent_path = self.dataset_path / scene / part_name

        if not self.agent_path.exists():
            raise FileNotFoundError(f"Agent path not found: {self.agent_path}")

        # Load color/depth paths
        results_dir = self.agent_path / "results"
        self.color_paths = sorted(results_dir.glob("frame*.jpg"))
        self.depth_paths = sorted(results_dir.glob("depth*.png"))

        # Load GT poses
        self.poses = []
        traj_path = self.agent_path / "traj.txt"
        with open(traj_path, "r") as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                c2w = np.array(vals).reshape(4, 4).astype(np.float32)
                self.poses.append(c2w)

        # Trim to min of color/depth/poses
        n = min(len(self.color_paths), len(self.depth_paths), len(self.poses))
        if self.frame_limit > 0:
            n = min(n, self.frame_limit)
        self.color_paths = self.color_paths[:n]
        self.depth_paths = self.depth_paths[:n]
        self.poses = self.poses[:n]

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        color = cv2.imread(str(self.color_paths[idx]))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0   # [H,W,3] float in [0,1]
        depth = cv2.imread(str(self.depth_paths[idx]), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / self.depth_scale  # [H,W] metres
        pose = self.poses[idx]
        return color, depth, pose


# ---------------------------------------------------------------------------
# Communication channel (simulation)
# ---------------------------------------------------------------------------

class AriaMultiagentLoader:
    """
    Loads RGB-D frames and GT poses for one agent from the AriaMultiagent dataset.

    Structure:
        {room}/agent_{id}/results/frame{:06d}.jpg
        {room}/agent_{id}/results/depth{:06d}.png
        {room}/agent_{id}/traj.txt
    """

    def __init__(self, dataset_path: str, scene: str, agent_id: int, config: dict):
        self.dataset_path = Path(dataset_path)
        self.scene = scene
        self.agent_id = agent_id
        self.depth_scale = config["dataset"]["depth_scale"]
        self.frame_limit = config.get("frame_limit", -1)

        self.agent_path = self.dataset_path / scene / f"agent_{agent_id}"

        if not self.agent_path.exists():
            raise FileNotFoundError(f"Agent path not found: {self.agent_path}")

        # Load color/depth paths
        results_dir = self.agent_path / "results"
        self.color_paths = sorted(results_dir.glob("frame*.jpg"))
        self.depth_paths = sorted(results_dir.glob("depth*.png"))

        # Load GT poses (4x4 c2w matrices, 16 floats per line)
        self.poses = []
        traj_path = self.agent_path / "traj.txt"
        with open(traj_path, "r") as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                c2w = np.array(vals).reshape(4, 4).astype(np.float32)
                self.poses.append(c2w)

        # Trim to min of color/depth/poses
        n = min(len(self.color_paths), len(self.depth_paths), len(self.poses))
        if self.frame_limit > 0:
            n = min(n, self.frame_limit)
        self.color_paths = self.color_paths[:n]
        self.depth_paths = self.depth_paths[:n]
        self.poses = self.poses[:n]

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        color = cv2.imread(str(self.color_paths[idx]))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0   # [H,W,3] float in [0,1]
        depth = cv2.imread(str(self.depth_paths[idx]), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / self.depth_scale  # [H,W] metres
        pose = self.poses[idx]
        return color, depth, pose


class SimChannel:
    """
    Simulates inter-robot communication for single-machine experiments.
    All RobotNode instances share one SimChannel object.
    """

    def __init__(self, bandwidth_bps: float = float('inf'), loss_rate: float = 0.0):
        self.bandwidth_bps = bandwidth_bps
        self.loss_rate = loss_rate
        self._queues = defaultdict(list)
        self._stats = {"sent": 0, "dropped": 0, "total_bytes": 0}

    def send(self, msg: RobotMessage) -> bool:
        if np.random.random() < self.loss_rate:
            self._stats["dropped"] += 1
            return False
        msg.byte_size = msg.byte_size or len(msg.payload)
        self._queues[msg.receiver_id].append(msg)
        self._stats["sent"] += 1
        self._stats["total_bytes"] += msg.byte_size
        return True

    def receive(self, robot_id: int):
        msgs = self._queues[robot_id]
        self._queues[robot_id] = []
        return msgs

    @property
    def stats(self):
        return self._stats


# ---------------------------------------------------------------------------
# Overlap-based loop closure detection
# ---------------------------------------------------------------------------

def detect_loop_closures(poses, overlap_thresh=0.3, distance_thresh=2.0):
    """
    Detect potential inter-robot loop closures based on pose proximity.
    Supports N agents. Returns list of (frame_idx, agent_i, agent_j, T_rel) tuples.
    """
    n_agents = len(poses)
    closures = []
    for ai in range(n_agents):
        for aj in range(ai + 1, n_agents):
            n_min = min(len(poses[ai]), len(poses[aj]))
            step = max(1, n_min // 20)
            for k in range(0, n_min, step):
                T_i = torch.from_numpy(poses[ai][k]).float()
                T_j = torch.from_numpy(poses[aj][k]).float()
                dist = (T_i[:3, 3] - T_j[:3, 3]).norm().item()
                if dist < distance_thresh:
                    T_rel = torch.linalg.inv(T_i) @ T_j
                    closures.append((k, ai, aj, T_rel))
    return closures


# ---------------------------------------------------------------------------
# Trajectory saving (TUM format)
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (qx, qy, qz, qw)."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def save_trajectory_tum(poses_list, timestamps, filepath):
    """Save trajectory in TUM format: timestamp tx ty tz qx qy qz qw"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for t, pose in zip(timestamps, poses_list):
            if isinstance(pose, torch.Tensor):
                pose = pose.detach().cpu().numpy()
            tx, ty, tz = pose[:3, 3]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(pose[:3, :3])
            f.write(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def save_pointcloud_ply(means, filepath):
    """Save Gaussian means as a PLY point cloud."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if isinstance(means, torch.Tensor):
        means = means.detach().cpu().numpy()
    N = means.shape[0]
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{means[i, 0]:.6f} {means[i, 1]:.6f} {means[i, 2]:.6f}\n")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ate_rmse(gt_poses, est_poses):
    """Compute Absolute Trajectory Error (RMSE) in meters with SE(3) alignment."""
    n = min(len(gt_poses), len(est_poses))
    if n < 3:
        return float('nan')

    # Extract camera positions (c2w translation column)
    gt_pos = np.zeros((n, 3))
    est_pos = np.zeros((n, 3))
    for i in range(n):
        gt = gt_poses[i] if isinstance(gt_poses[i], np.ndarray) else gt_poses[i].detach().cpu().numpy()
        est = est_poses[i] if isinstance(est_poses[i], np.ndarray) else est_poses[i].detach().cpu().numpy()
        gt_pos[i] = gt[:3, 3]
        est_pos[i] = est[:3, 3]

    # SE(3) Umeyama alignment: find R, t so that R @ est + t ≈ gt
    gt_mean = gt_pos.mean(axis=0)
    est_mean = est_pos.mean(axis=0)
    gt_c = gt_pos - gt_mean
    est_c = est_pos - est_mean
    H = est_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = gt_mean - R @ est_mean
    est_aligned = (R @ est_pos.T).T + t

    errors = np.linalg.norm(gt_pos - est_aligned, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_rpe(gt_poses, est_poses):
    """Compute Relative Pose Error (translation and rotation)."""
    n = min(len(gt_poses), len(est_poses))
    trans_errors = []
    rot_errors = []
    for i in range(1, n):
        gt_prev = gt_poses[i - 1] if isinstance(gt_poses[i - 1], np.ndarray) else gt_poses[i - 1].detach().cpu().numpy()
        gt_curr = gt_poses[i] if isinstance(gt_poses[i], np.ndarray) else gt_poses[i].detach().cpu().numpy()
        est_prev = est_poses[i - 1] if isinstance(est_poses[i - 1], np.ndarray) else est_poses[i - 1].detach().cpu().numpy()
        est_curr = est_poses[i] if isinstance(est_poses[i], np.ndarray) else est_poses[i].detach().cpu().numpy()

        gt_rel = np.linalg.inv(gt_prev) @ gt_curr
        est_rel = np.linalg.inv(est_prev) @ est_curr
        err_rel = np.linalg.inv(gt_rel) @ est_rel

        trans_errors.append(np.linalg.norm(err_rel[:3, 3]))
        # Rotation error in degrees
        cos_angle = np.clip((np.trace(err_rel[:3, :3]) - 1) / 2, -1, 1)
        rot_errors.append(np.degrees(np.arccos(cos_angle)))

    return float(np.mean(trans_errors)), float(np.mean(rot_errors))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment_name"]
    dataset_cfg = config["dataset"]
    system_cfg = config["system"]
    output_dir = config.get("output", {}).get("dir", "outputs")
    seed = config.get("seed", 42)
    frame_limit = config.get("frame_limit", -1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = system_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    scenes = dataset_cfg["scenes"]
    n_agents = dataset_cfg["n_agents"]

    all_results = {}

    for scene in scenes:
        print(f"\n{'='*60}")
        print(f"Scene: {scene}")
        print(f"{'='*60}")

        # Select loader class based on dataset name
        dataset_name = dataset_cfg.get("name", "replica_multiagent")
        if dataset_name == "aria_multiagent":
            LoaderClass = AriaMultiagentLoader
        else:
            LoaderClass = ReplicaMultiagentLoader

        # Load dataset for each agent
        loaders = []
        for agent_id in range(n_agents):
            try:
                loader = LoaderClass(
                    dataset_path=dataset_cfg["path"],
                    scene=scene,
                    agent_id=agent_id,
                    config=config,
                )
                loaders.append(loader)
                print(f"  Agent {agent_id}: {len(loader)} frames loaded")
            except FileNotFoundError as e:
                print(f"  Agent {agent_id}: SKIPPED ({e})")
                loaders.append(None)

        if any(l is None for l in loaders):
            print(f"  Skipping scene {scene} — not all agents have data")
            continue

        n_frames = min(len(l) for l in loaders)
        if frame_limit > 0:
            n_frames = min(n_frames, frame_limit)
        print(f"  Processing {n_frames} frames per agent")

        # Create communication channel
        channel = SimChannel()

        # Load MonoGS config for LocalSLAMWrapper
        monogs_config_path = config.get("monogs_config_path")
        monogs_config = None
        if monogs_config_path:
            with open(os.path.join(str(PROJECT_ROOT), monogs_config_path)) as mf:
                monogs_config = yaml.safe_load(mf)

        # Create robot nodes
        robot_config = {
            "device": device,
            "n_init_gaussians": system_cfg.get("n_init_gaussians", 500),
            "k_fim": system_cfg.get("fim_window_keyframes", 10),
            "match_distance_thresh": system_cfg.get("gaussian_match_thresh", 0.05),
            "match_opacity_thresh": 0.1,
            "rho_init": system_cfg.get("admm_rho_init", 1.0),
            "rho_max": system_cfg.get("admm_rho_max", 100.0),
            "pose_lr": system_cfg.get("pose_lr", 0.1),
            "tol_primal": system_cfg.get("admm_tol_primal", 1e-4),
            "tol_dual": system_cfg.get("admm_tol_dual", 1e-4),
            "use_fim_weighting": system_cfg.get("use_fim_weighting", True),
        }

        # Pass camera intrinsics for FIM visibility check
        if monogs_config is not None:
            cal = monogs_config.get("Dataset", {}).get("Calibration", {})
            if "fx" in cal:
                robot_config["fx"] = cal["fx"]
                robot_config["fy"] = cal["fy"]
                robot_config["cx"] = cal["cx"]
                robot_config["cy"] = cal["cy"]
                robot_config["H"] = cal.get("height", 680)
                robot_config["W"] = cal.get("width", 1200)

        robots = []
        slam_wrappers = []
        for agent_id in range(n_agents):
            slam_step = None
            wrapper = None
            if monogs_config is not None:
                import copy
                mc = copy.deepcopy(monogs_config)
                # Inject max_gaussians from experiment config into MonoGS config
                mc.setdefault("Training", {})["max_gaussians"] = system_cfg.get("max_gaussians", 80000)
                wrapper = LocalSLAMWrapper(config=mc, device=device)
                slam_step = wrapper.as_slam_step()

            robot = RobotNode(
                robot_id=agent_id,
                config=robot_config,
                comm_channel=channel,
                slam_step=slam_step,
            )
            robots.append(robot)
            slam_wrappers.append(wrapper)

        # Collect GT poses and estimated poses for metrics
        gt_trajectories = [[] for _ in range(n_agents)]
        est_trajectories = [[] for _ in range(n_agents)]
        timestamps = []

        # Pre-detect loop closures from GT poses
        gt_poses_all = [loaders[a].poses for a in range(n_agents)]
        loop_closures = detect_loop_closures(gt_poses_all, distance_thresh=2.0)
        loop_closure_injected = set()

        print(f"  Detected {len(loop_closures)} potential loop closures")

        t_start = time.time()

        # Main processing loop
        for t in range(n_frames):
            timestamp = float(t) / 30.0  # assume 30 fps

            for agent_id in range(n_agents):
                color, depth, gt_pose = loaders[agent_id][t]

                # Convert to tensors
                rgb_tensor = torch.from_numpy(color).float()
                depth_tensor = torch.from_numpy(depth).float()

                # Feed frame to robot (LocalSLAMWrapper produces real poses)
                robots[agent_id].process_frame(rgb_tensor, depth_tensor, timestamp)

                # Record trajectories (GT is c2w, SLAM output is w2c — invert)
                gt_trajectories[agent_id].append(gt_pose)
                est_w2c = robots[agent_id].current_pose.detach().cpu().numpy()
                est_c2w = np.linalg.inv(est_w2c)
                est_trajectories[agent_id].append(est_c2w)

            timestamps.append(timestamp)

            # Inject loop closures at detected frames
            for lc_frame, ai, aj, T_rel in loop_closures:
                lc_key = (lc_frame, ai, aj)
                if lc_frame == t and lc_key not in loop_closure_injected:
                    info = torch.eye(6, device=device) * 100.0
                    robots[ai].add_loop_closure(
                        neighbor_id=aj,
                        relative_pose=T_rel.to(device),
                        info=info,
                    )
                    robots[aj].add_loop_closure(
                        neighbor_id=ai,
                        relative_pose=torch.linalg.inv(T_rel).to(device),
                        info=info,
                    )
                    # Share poses for ADMM
                    robots[ai]._neighbor_poses[aj] = robots[aj].current_pose.clone()
                    robots[aj]._neighbor_poses[ai] = robots[ai].current_pose.clone()
                    loop_closure_injected.add(lc_key)

            # Progress reporting and memory management
            if (t + 1) % 100 == 0 or t == n_frames - 1:
                elapsed = time.time() - t_start
                fps = (t + 1) / elapsed
                gpu_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                n_gauss = [w.gaussian_model._xyz.shape[0] if w and w.gaussian_model is not None else 0 for w in slam_wrappers]
                print(f"  Frame {t+1}/{n_frames} ({fps:.1f} fps, GPU={gpu_mb:.0f}MB, Gaussians={n_gauss})")
                gc.collect()
                torch.cuda.empty_cache()

        elapsed_total = time.time() - t_start
        print(f"  Done in {elapsed_total:.1f}s")

        # Compute metrics
        scene_results = {"scene": scene, "n_frames": n_frames, "elapsed_s": elapsed_total}

        for agent_id in range(n_agents):
            ate = compute_ate_rmse(gt_trajectories[agent_id], est_trajectories[agent_id])
            rpe_t, rpe_r = compute_rpe(gt_trajectories[agent_id], est_trajectories[agent_id])
            scene_results[f"agent_{agent_id}_ATE_RMSE"] = ate
            scene_results[f"agent_{agent_id}_RPE_trans"] = rpe_t
            scene_results[f"agent_{agent_id}_RPE_rot_deg"] = rpe_r
            print(f"  Agent {agent_id}: ATE={ate:.4f}m, RPE_t={rpe_t:.4f}m, RPE_r={rpe_r:.2f}°")

        # ADMM stats
        for agent_id in range(n_agents):
            monitor = robots[agent_id].convergence_monitor
            summary = monitor.summary()
            scene_results[f"agent_{agent_id}_admm_iterations"] = summary["n_iterations"]
            scene_results[f"agent_{agent_id}_last_primal"] = summary["last_primal"]

        scene_results["comm_bytes_total"] = channel.stats["total_bytes"]
        scene_results["comm_sent"] = channel.stats["sent"]
        scene_results["comm_dropped"] = channel.stats["dropped"]
        scene_results["use_fim_weighting"] = system_cfg.get("use_fim_weighting", True)

        # Save trajectories
        if config.get("output", {}).get("save_trajectory", True):
            for agent_id in range(n_agents):
                traj_dir = os.path.join(output_dir, "trajectories", experiment_name)
                save_trajectory_tum(
                    est_trajectories[agent_id], timestamps,
                    os.path.join(traj_dir, f"robot_{agent_id}_est.txt"),
                )
                save_trajectory_tum(
                    gt_trajectories[agent_id], timestamps,
                    os.path.join(traj_dir, f"robot_{agent_id}_gt.txt"),
                )

        # Save map
        if config.get("output", {}).get("save_map", True):
            for agent_id in range(n_agents):
                map_dir = os.path.join(output_dir, "maps", experiment_name)
                save_pointcloud_ply(
                    robots[agent_id].gaussian_map.means,
                    os.path.join(map_dir, f"robot_{agent_id}_{scene}_final.ply"),
                )

        all_results[scene] = scene_results

        # Compute ECE (Expected Calibration Error) for H3 validation.
        # Uses FIM from SLAM and backprojected GT surface from last few frames.
        for agent_id in range(n_agents):
            gmap = robots[agent_id].gaussian_map
            if gmap.fim_means is None:
                continue

            fim_trace = gmap.fim_means.sum(dim=-1).cpu().numpy()  # [N]
            means_np = gmap.means.detach().cpu().numpy()  # [N, 3]
            N = len(fim_trace)
            if N < 10:
                continue

            # Build GT surface cloud from last 5 GT frames (local evaluation)
            cam_cfg = dataset_cfg.get("cam", {})
            fx = cam_cfg.get("fx", 280.0)
            fy = cam_cfg.get("fy", 280.0)
            cx = cam_cfg.get("cx", 255.5)
            cy = cam_cfg.get("cy", 255.5)

            gt_points_all = []
            for fi in range(max(0, n_frames - 5), n_frames):
                _, depth_frame, gt_pose = loaders[agent_id][fi]
                H_img, W_img = depth_frame.shape[:2]
                u, v = np.meshgrid(np.arange(W_img), np.arange(H_img))
                z = depth_frame.reshape(H_img, W_img)
                valid = z > 0.01
                x3d = (u[valid] - cx) * z[valid] / fx
                y3d = (v[valid] - cy) * z[valid] / fy
                z3d = z[valid]
                pts_cam = np.stack([x3d, y3d, z3d], axis=-1)
                R_gt = gt_pose[:3, :3]
                t_gt = gt_pose[:3, 3]
                pts_world = (R_gt @ pts_cam.T).T + t_gt
                if len(pts_world) > 20000:
                    idx = np.random.choice(len(pts_world), 20000, replace=False)
                    pts_world = pts_world[idx]
                gt_points_all.append(pts_world)

            gt_cloud = np.concatenate(gt_points_all, axis=0).astype(np.float32)

            # Nearest-surface distance per Gaussian (no alignment — both in SLAM frame)
            from scipy.spatial import cKDTree
            tree = cKDTree(gt_cloud)
            dists, _ = tree.query(means_np, k=1)
            recon_errors = dists.astype(np.float32)

            # ECE with quantile-based binning
            n_bins = 10

            # Predicted reliability via rank percentile of FIM
            rank_order = np.argsort(fim_trace)
            predicted_reliability = np.zeros(N)
            predicted_reliability[rank_order] = np.linspace(0, 1, N)

            # Actual reliability: 1 - normalized error (95th percentile robust)
            e_p95 = np.percentile(recon_errors, 95)
            error_clipped = np.clip(recon_errors, 0, e_p95)
            actual_reliability = 1.0 - error_clipped / (e_p95 + 1e-10)
            actual_reliability = np.clip(actual_reliability, 0, 1)

            # Quantile-based binning
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_idx = np.digitize(predicted_reliability, bin_edges[1:-1])

            ece = 0.0
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() == 0:
                    continue
                pred_mean = predicted_reliability[mask].mean()
                actual_mean = actual_reliability[mask].mean()
                ece += (mask.sum() / N) * abs(actual_mean - pred_mean)

            scene_results[f"agent_{agent_id}_ECE"] = float(ece)
            print(f"  Agent {agent_id}: ECE = {ece:.4f} (H3 target: < 0.05)")

            # Spearman rank correlation
            from scipy.stats import spearmanr
            corr, pval = spearmanr(fim_trace, -recon_errors)
            scene_results[f"agent_{agent_id}_FIM_error_spearman"] = float(corr)
            print(f"  Agent {agent_id}: FIM↔error Spearman r = {corr:.4f} (p={pval:.2e})")

            # Save for calibration plot
            ece_dir = os.path.join(output_dir, "ece", experiment_name)
            os.makedirs(ece_dir, exist_ok=True)
            np.save(os.path.join(ece_dir, f"{scene}_agent_{agent_id}_fim_traces.npy"), fim_trace)
            np.save(os.path.join(ece_dir, f"{scene}_agent_{agent_id}_recon_errors.npy"), recon_errors)

    # Save combined results
    if config.get("output", {}).get("save_metrics", True):
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{experiment_name}.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run DUAG-C experiment")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
