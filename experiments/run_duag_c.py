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
import re
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


class S3ELoader:
    """
    Loads RGB-D frames and GT poses for one robot from S3Ev1 dataset.

    Structure:
        S3Ev1/{sequence}/{Robot}/Camera/image_left/{timestamp}.jpg
        S3Ev1/{sequence}/{Robot}/Camera/depth/{timestamp}.png   (stereo-computed)
        S3Ev1/{sequence}/{robot}_gt.txt   (timestamp x y z qx qy qz qw, ~1Hz GPS)
        S3Ev1/Calibration/{robot}.yaml

    GT is at ~1Hz, images at 10Hz. Poses are interpolated to image timestamps.
    UTM coordinates are centered relative to the first pose.
    """

    ROBOT_NAMES = ["Alpha", "Bob", "Carol"]

    def __init__(self, dataset_path: str, scene: str, agent_id: int, config: dict):
        self.dataset_path = Path(dataset_path)
        self.scene = scene
        self.agent_id = agent_id
        self.depth_scale = config["dataset"].get("depth_scale", 1000.0)
        self.frame_limit = config.get("frame_limit", -1)
        self.frame_step = config.get("frame_step", 1)

        robot_name = self.ROBOT_NAMES[agent_id]
        self.robot_name = robot_name
        base = self.dataset_path / "S3Ev1"

        # Image directory
        self.img_dir = base / scene / robot_name / "Camera" / "image_left"
        self.depth_dir = base / scene / robot_name / "Camera" / "depth"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Load GT trajectory
        gt_path = base / scene / f"{robot_name.lower()}_gt.txt"
        if not gt_path.exists():
            raise FileNotFoundError(f"GT trajectory not found: {gt_path}")
        self._gt_timestamps, self._gt_poses_raw = self._load_gt(gt_path)
        if len(self._gt_timestamps) < 2:
            raise ValueError(f"GT has <2 poses for {scene}/{robot_name}: unusable")

        # Center GT positions relative to first pose
        origin = self._gt_poses_raw[0][:3, 3].copy()
        for p in self._gt_poses_raw:
            p[:3, 3] -= origin

        # Load image file list and extract timestamps
        all_imgs = sorted(self.img_dir.glob("*.jpg"))
        if not all_imgs:
            raise FileNotFoundError(f"No images in {self.img_dir}")

        # Only keep images within GT time coverage
        gt_t_min = self._gt_timestamps[0]
        gt_t_max = self._gt_timestamps[-1]
        self._img_paths = []
        self._img_timestamps = []
        for p in all_imgs:
            ts = float(p.stem)
            if gt_t_min <= ts <= gt_t_max:
                self._img_paths.append(p)
                self._img_timestamps.append(ts)

        # Apply frame_step (subsample)
        if self.frame_step > 1:
            self._img_paths = self._img_paths[::self.frame_step]
            self._img_timestamps = self._img_timestamps[::self.frame_step]

        # Apply frame_limit
        if self.frame_limit > 0:
            self._img_paths = self._img_paths[:self.frame_limit]
            self._img_timestamps = self._img_timestamps[:self.frame_limit]

        # Interpolate GT poses at image timestamps
        self.poses = self._interpolate_poses()

    def _load_gt(self, path):
        """Load TUM-format GT: timestamp x y z qx qy qz qw."""
        timestamps = []
        poses = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Handle missing spaces (e.g. "-0.041-0.296")
                parts = re.split(r'\s+', line)
                if len(parts) < 8:
                    continue
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                from scipy.spatial.transform import Rotation
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R.astype(np.float32)
                T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
                timestamps.append(ts)
                poses.append(T)
        return timestamps, poses

    def _interpolate_poses(self):
        """Interpolate GT poses (1Hz) at image timestamps (10Hz) using linear lerp."""
        gt_ts = np.array(self._gt_timestamps)
        poses_out = []
        for img_ts in self._img_timestamps:
            # Find bracketing GT poses
            idx = np.searchsorted(gt_ts, img_ts, side='right') - 1
            idx = max(0, min(idx, len(gt_ts) - 2))
            t0, t1 = gt_ts[idx], gt_ts[idx + 1]
            dt = t1 - t0
            if dt < 1e-6:
                alpha = 0.0
            else:
                alpha = float(np.clip((img_ts - t0) / dt, 0.0, 1.0))
            # Linear interpolation of position
            p0 = self._gt_poses_raw[idx]
            p1 = self._gt_poses_raw[idx + 1]
            pose = p0.copy()
            pose[:3, 3] = (1.0 - alpha) * p0[:3, 3] + alpha * p1[:3, 3]
            # For rotation: use nearest (GT has identity rotation from GPS)
            poses_out.append(pose)
        return poses_out

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        # Load RGB image
        color = cv2.imread(str(self._img_paths[idx]))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0  # [H,W,3]

        # Load stereo depth if available, else return zeros
        depth_path = self.depth_dir / f"{self._img_paths[idx].stem}.png"
        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / self.depth_scale  # mm -> metres
        else:
            H, W = color.shape[:2]
            depth = np.zeros((H, W), dtype=np.float32)

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


class KimeraMultiLoader:
    """
    Loads GT poses for one robot from the Kimera-Multi-Data Campus dataset.

    GT format: CSV with header  #timestamp_kf,x,y,z,qw,qx,qy,qz
    Timestamps are in nanoseconds, quaternion is w-first.
    No images available (locked in rosbags) — returns blank frames.
    Experiment runs in synthetic drift mode.

    Structure:
        {sequence}/gt/{robot_name}_gt_odom.csv
    """

    # Robot ID -> name mapping (Kimera-Multi convention)
    ROBOT_NAMES = {
        0: "acl_jackal",
        1: "acl_jackal2",
        2: "sparkal1",
        3: "sparkal2",
        4: "hathor",
        5: "thoth",
        6: "apis",
        7: "sobek",
    }

    def __init__(self, dataset_path: str, scene: str, agent_id: int, config: dict):
        self.dataset_path = Path(dataset_path)
        self.scene = scene
        self.agent_id = agent_id
        self.frame_limit = config.get("frame_limit", -1)
        self.frame_step = config.get("frame_step", 1)

        robot_name = self.ROBOT_NAMES[agent_id]
        gt_path = self.dataset_path / scene / "gt" / f"{robot_name}_gt_odom.csv"
        if not gt_path.exists():
            raise FileNotFoundError(f"GT not found: {gt_path}")

        # Parse CSV: timestamp_ns,x,y,z,qw,qx,qy,qz
        data = np.loadtxt(str(gt_path), delimiter=',', skiprows=1)
        timestamps_ns = data[:, 0]
        self._timestamps = timestamps_ns * 1e-9  # -> seconds

        from scipy.spatial.transform import Rotation
        self.poses = []
        for i in range(len(data)):
            x, y, z = data[i, 1], data[i, 2], data[i, 3]
            qw, qx, qy, qz = data[i, 4], data[i, 5], data[i, 6], data[i, 7]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R.astype(np.float32)
            T[:3, 3] = np.array([x, y, z], dtype=np.float32)
            self.poses.append(T)

        # Center positions relative to first pose
        origin = self.poses[0][:3, 3].copy()
        for p in self.poses:
            p[:3, 3] -= origin

        # Subsample
        if self.frame_step > 1:
            self.poses = self.poses[::self.frame_step]
            self._timestamps = self._timestamps[::self.frame_step]

        # Apply frame limit
        if self.frame_limit > 0:
            self.poses = self.poses[:self.frame_limit]
            self._timestamps = self._timestamps[:self.frame_limit]

        # Camera resolution from config (used for dummy images)
        cam = config.get("dataset", {}).get("cam", {})
        self._H = cam.get("H", 480)
        self._W = cam.get("W", 640)

    @property
    def _img_timestamps(self):
        return self._timestamps.tolist()

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        # No images available — return blank frames for synthetic mode
        color = np.zeros((self._H, self._W, 3), dtype=np.float32)
        depth = np.zeros((self._H, self._W), dtype=np.float32)
        pose = self.poses[idx]
        return color, depth, pose


class SimChannel:
    """
    Simulates inter-robot communication for single-machine experiments.
    All RobotNode instances share one SimChannel object.

    bandwidth_bytes_per_frame: max bytes each sender can transmit per frame.
        Default inf = unlimited. Call reset_frame_budget() once per frame.
    """

    def __init__(self, bandwidth_bytes_per_frame: float = float('inf'), loss_rate: float = 0.0):
        self.bandwidth_bytes_per_frame = bandwidth_bytes_per_frame
        self.loss_rate = loss_rate
        self._queues = defaultdict(list)
        self._bytes_sent_this_frame = defaultdict(int)
        self._stats = {"sent": 0, "dropped": 0, "dropped_bw": 0, "total_bytes": 0}

    def send(self, msg: RobotMessage) -> bool:
        if np.random.random() < self.loss_rate:
            self._stats["dropped"] += 1
            return False
        msg.byte_size = msg.byte_size or len(msg.payload)
        # Enforce per-frame bandwidth budget per sender
        if self.bandwidth_bytes_per_frame < float('inf'):
            if self._bytes_sent_this_frame[msg.sender_id] + msg.byte_size > self.bandwidth_bytes_per_frame:
                self._stats["dropped"] += 1
                self._stats["dropped_bw"] += 1
                return False
        self._bytes_sent_this_frame[msg.sender_id] += msg.byte_size
        self._queues[msg.receiver_id].append(msg)
        self._stats["sent"] += 1
        self._stats["total_bytes"] += msg.byte_size
        return True

    def receive(self, robot_id: int):
        msgs = self._queues[robot_id]
        self._queues[robot_id] = []
        return msgs

    def reset_frame_budget(self):
        """Call once per frame to reset the per-sender bandwidth counter."""
        self._bytes_sent_this_frame = defaultdict(int)

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
# Config loading with inherit_from support (matches MAGiC-SLAM / MonoGS pattern)
# ---------------------------------------------------------------------------

def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str) -> dict:
    """
    Load a YAML config with optional ``inherit_from`` base config.

    If the config contains an ``inherit_from`` key, load the base config first
    (resolving its path relative to the child config's directory), then
    recursively merge the child's values on top.  Supports one level of
    inheritance (same as MAGiC-SLAM / MonoGS).
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "inherit_from" in config:
        base_path = config.pop("inherit_from")
        # Resolve relative to child config's directory
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_config = load_config(base_path)   # recursive — supports chaining
        config = _merge_dicts(base_config, config)

    return config


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(config_path: str, bandwidth_bytes_per_frame: float = float('inf'),
                   loss_rate: float = 0.0):
    config = load_config(config_path)

    experiment_name = config["experiment_name"]
    # Tag experiment name with comm constraint setting
    if bandwidth_bytes_per_frame < float('inf'):
        bw_tag = f"bw_{int(bandwidth_bytes_per_frame)}"
        experiment_name = f"{experiment_name}_{bw_tag}"
        config["experiment_name"] = experiment_name
    if loss_rate > 0:
        lr_tag = f"loss_{int(loss_rate * 100)}pct"
        experiment_name = f"{experiment_name}_{lr_tag}"
        config["experiment_name"] = experiment_name
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
        elif dataset_name == "s3e":
            LoaderClass = S3ELoader
        elif dataset_name == "kimera_campus":
            LoaderClass = KimeraMultiLoader
        else:
            LoaderClass = ReplicaMultiagentLoader

        # Load dataset for each agent
        # _robot_id_map remaps agent indices to real robot IDs (for Kimera subsets)
        robot_id_map = dataset_cfg.get("_robot_id_map", list(range(n_agents)))
        loaders = []
        for agent_id in range(n_agents):
            real_robot_id = robot_id_map[agent_id]
            try:
                loader = LoaderClass(
                    dataset_path=dataset_cfg["path"],
                    scene=scene,
                    agent_id=real_robot_id,
                    config=config,
                )
                loaders.append(loader)
                print(f"  Agent {agent_id} (robot {real_robot_id}): {len(loader)} frames loaded")
            except FileNotFoundError as e:
                print(f"  Agent {agent_id} (robot {real_robot_id}): SKIPPED ({e})")
                loaders.append(None)

        if any(l is None for l in loaders):
            print(f"  Skipping scene {scene} — not all agents have data")
            continue

        n_frames = min(len(l) for l in loaders)
        if frame_limit > 0:
            n_frames = min(n_frames, frame_limit)
        print(f"  Processing {n_frames} frames per agent")

        # Create communication channel (with optional bandwidth throttling / loss)
        channel = SimChannel(
            bandwidth_bytes_per_frame=bandwidth_bytes_per_frame,
            loss_rate=loss_rate,
        )

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
        elif "cam" in dataset_cfg:
            cam = dataset_cfg["cam"]
            robot_config["fx"] = cam["fx"]
            robot_config["fy"] = cam["fy"]
            robot_config["cx"] = cam["cx"]
            robot_config["cy"] = cam["cy"]
            robot_config["H"] = cam.get("H", 1024)
            robot_config["W"] = cam.get("W", 1224)

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
        loop_closures = detect_loop_closures(
            gt_poses_all,
            distance_thresh=system_cfg.get("loop_closure_distance_thresh", 2.0),
        )
        loop_closure_injected = set()

        print(f"  Detected {len(loop_closures)} potential loop closures")

        t_start = time.time()

        # Initialize per-agent cumulative drift for synthetic mode
        drift_poses = [torch.eye(4, device=device) for _ in range(n_agents)]

        # Main processing loop
        for t in range(n_frames):
            # Use real timestamps from loader if available (S3E), else synthesize
            if hasattr(loaders[0], '_img_timestamps'):
                timestamp = loaders[0]._img_timestamps[t]
            else:
                timestamp = float(t) / 30.0  # assume 30 fps

            for agent_id in range(n_agents):
                color, depth, gt_pose = loaders[agent_id][t]

                # Convert to tensors
                rgb_tensor = torch.from_numpy(color).float()
                depth_tensor = torch.from_numpy(depth).float()

                # For synthetic mode (no MonoGS): use GT pose + cumulative drift
                # Random walk drift simulates odometry error accumulation.
                # Loop closures should correct this drift, showing consensus value.
                if slam_wrappers[agent_id] is None:
                    gt_pose_t = torch.from_numpy(gt_pose).float().to(device)
                    # Accumulate small per-frame drift (random walk)
                    drift_step = torch.eye(4, device=device)
                    drift_step[:3, 3] = torch.randn(3, device=device) * 0.01  # 1cm/frame std
                    drift_poses[agent_id] = drift_poses[agent_id] @ drift_step
                    # Apply accumulated drift to GT pose
                    noisy_pose = gt_pose_t @ drift_poses[agent_id]
                    # Set as w2c (invert GT c2w)
                    robots[agent_id].current_pose = torch.linalg.inv(noisy_pose)

                # Feed frame to robot (LocalSLAMWrapper produces real poses)
                robots[agent_id].process_frame(rgb_tensor, depth_tensor, timestamp)

                # Record trajectories (GT is c2w, SLAM output is w2c — invert)
                gt_trajectories[agent_id].append(gt_pose)
                est_w2c = robots[agent_id].current_pose.detach().cpu().numpy()
                est_c2w = np.linalg.inv(est_w2c)
                est_trajectories[agent_id].append(est_c2w)

            timestamps.append(timestamp)

            # Reset per-frame bandwidth budget
            channel.reset_frame_budget()

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
        scene_results["comm_dropped_bw"] = channel.stats.get("dropped_bw", 0)
        scene_results["bandwidth_bytes_per_frame"] = bandwidth_bytes_per_frame
        scene_results["use_fim_weighting"] = system_cfg.get("use_fim_weighting", True)

        # Optional post-experiment map refinement
        refine_iters = system_cfg.get("refine_iterations", 0)
        if refine_iters > 0:
            for agent_id in range(n_agents):
                w = slam_wrappers[agent_id]
                if w is not None:
                    print(f"  Refining agent {agent_id} map ({refine_iters} iters)...")
                    w.refine_map(iters=refine_iters)

        # Render quality metrics (PSNR, SSIM, LPIPS, DepthL1) on keyframes
        for agent_id in range(n_agents):
            w = slam_wrappers[agent_id]
            if w is not None:
                rm = w.compute_render_metrics()
                for k, v in rm.items():
                    scene_results[f"agent_{agent_id}_{k}"] = v
                print(f"  Agent {agent_id}: PSNR={rm['PSNR']:.2f}, SSIM={rm['SSIM']:.4f}, "
                      f"LPIPS={rm['LPIPS']:.4f}, DepthL1={rm['DepthL1']:.4f}")

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
        # Frustum-based FIM from SLAM + backprojected GT surface in SLAM frame.
        all_fim_traces = []
        all_recon_errors = []
        for agent_id in range(n_agents):
            gmap = robots[agent_id].gaussian_map
            if gmap.fim_means is None:
                continue

            fim_trace = gmap.fim_means.sum(dim=-1).cpu().numpy()  # [N]
            means_np = gmap.means.detach().cpu().numpy()  # [N, 3]
            N = len(fim_trace)
            if N < 10:
                continue

            # Camera intrinsics for backprojection
            cam_cfg = dataset_cfg.get("cam", {})
            fx = cam_cfg.get("fx", 280.0)
            fy = cam_cfg.get("fy", 280.0)
            cx = cam_cfg.get("cx", 255.5)
            cy = cam_cfg.get("cy", 255.5)

            # First GT pose — transforms GT world frame to SLAM frame
            _, _, gt_pose_0 = loaders[agent_id][0]
            gt0_inv = np.linalg.inv(gt_pose_0)

            # Build GT surface from every 10th frame across full trajectory
            gt_points_all = []
            for fi in range(0, n_frames, 10):
                _, depth_frame, gt_pose = loaders[agent_id][fi]
                H_img, W_img = depth_frame.shape[:2]
                u, v = np.meshgrid(np.arange(W_img), np.arange(H_img))
                z = depth_frame.reshape(H_img, W_img)
                valid = z > 0.01
                x3d = (u[valid] - cx) * z[valid] / fx
                y3d = (v[valid] - cy) * z[valid] / fy
                z3d = z[valid]
                pts_cam = np.stack([x3d, y3d, z3d], axis=-1)
                # Transform to SLAM frame: inv(gt_pose_0) @ gt_pose[fi]
                rel_pose = gt0_inv @ gt_pose
                R_rel = rel_pose[:3, :3]
                t_rel = rel_pose[:3, 3]
                pts_slam = (R_rel @ pts_cam.T).T + t_rel
                if len(pts_slam) > 20000:
                    idx = np.random.choice(len(pts_slam), 20000, replace=False)
                    pts_slam = pts_slam[idx]
                gt_points_all.append(pts_slam)

            gt_cloud = np.concatenate(gt_points_all, axis=0).astype(np.float32)

            # Nearest-surface distance per Gaussian
            from scipy.spatial import cKDTree
            tree = cKDTree(gt_cloud)
            dists, _ = tree.query(means_np, k=1)
            recon_errors = dists.astype(np.float32)

            # ECE using instruction document formula:
            # uncertainty = 1/sqrt(FIM), min-max normalize, value-based binning
            n_bins = 10
            uncertainty = 1.0 / np.sqrt(fim_trace + 1e-10)
            u_min, u_max = uncertainty.min(), uncertainty.max()
            u_range = u_max - u_min if (u_max - u_min) > 1e-10 else 1.0
            uncertainty_norm = (uncertainty - u_min) / u_range

            e_min, e_max = recon_errors.min(), recon_errors.max()
            e_range = e_max - e_min if (e_max - e_min) > 1e-10 else 1.0
            error_norm = (recon_errors - e_min) / e_range

            bin_idx = np.floor(uncertainty_norm * n_bins).clip(0, n_bins - 1).astype(int)

            ece = 0.0
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.sum() == 0:
                    continue
                pred_mean = uncertainty_norm[mask].mean()
                actual_mean = error_norm[mask].mean()
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

            # Collect for pooled ECE
            all_fim_traces.append(fim_trace)
            all_recon_errors.append(recon_errors)

        # Pooled ECE across all agents (standard calibration metric)
        if len(all_fim_traces) >= 2:
            fim_pool = np.concatenate(all_fim_traces)
            err_pool = np.concatenate(all_recon_errors)
            N_pool = len(fim_pool)
            n_bins = 10
            unc_pool = 1.0 / np.sqrt(fim_pool + 1e-10)
            u_min, u_max = unc_pool.min(), unc_pool.max()
            u_rng = u_max - u_min if (u_max - u_min) > 1e-10 else 1.0
            unc_pool_n = (unc_pool - u_min) / u_rng
            e_min, e_max = err_pool.min(), err_pool.max()
            e_rng = e_max - e_min if (e_max - e_min) > 1e-10 else 1.0
            err_pool_n = (err_pool - e_min) / e_rng
            bin_idx_p = np.floor(unc_pool_n * n_bins).clip(0, n_bins - 1).astype(int)
            ece_pooled = 0.0
            for b in range(n_bins):
                mask = bin_idx_p == b
                if mask.sum() == 0:
                    continue
                ece_pooled += (mask.sum() / N_pool) * abs(err_pool_n[mask].mean() - unc_pool_n[mask].mean())
            scene_results["ECE_pooled"] = float(ece_pooled)
            print(f"  Pooled ECE = {ece_pooled:.4f} (H3 target: < 0.05, N={N_pool})")

    # Save combined results
    if config.get("output", {}).get("save_metrics", True):
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{experiment_name}.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")

    return all_results


def run_kimera_subsets(config_path: str):
    """Run Kimera-Multi experiment with multiple robot subsets for N-robot scaling."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    robot_subsets = config["dataset"].get("robot_subsets", [[0, 1]])
    base_name = config["experiment_name"]
    all_subset_results = {}

    for subset in robot_subsets:
        n = len(subset)
        print(f"\n{'#'*60}")
        print(f"Running with {n}-robot subset: {subset}")
        print(f"{'#'*60}")

        # Update config for this subset
        config["experiment_name"] = f"{base_name}_{n}robots"
        config["dataset"]["n_agents"] = n
        # Remap robot IDs: subset [0,2,4] -> loader agent_ids 0,2,4
        config["dataset"]["_robot_id_map"] = subset

        # Write temp config
        tmp_path = config_path.replace(".yaml", f"_tmp_{n}robots.yaml")
        with open(tmp_path, "w") as f:
            yaml.dump(config, f)

        results = run_experiment(tmp_path)
        all_subset_results[f"{n}_robots"] = results

        # Clean up
        os.remove(tmp_path)

    # Print N-robot scaling summary
    print(f"\n{'='*60}")
    print("N-robot scaling summary")
    print(f"{'='*60}")
    for key, res in sorted(all_subset_results.items()):
        for scene, sr in res.items():
            ates = [v for k, v in sr.items() if "ATE_RMSE" in k and isinstance(v, float)]
            mean_ate = float(np.nanmean(ates)) if ates else float('nan')
            print(f"  {key} / {scene}: mean ATE = {mean_ate:.4f}m")

    # Save scaling results
    output_dir = config.get("output", {}).get("dir", "outputs")
    scaling_path = os.path.join(output_dir, "results", f"{base_name}_scaling.json")
    os.makedirs(os.path.dirname(scaling_path), exist_ok=True)
    with open(scaling_path, "w") as f:
        json.dump(all_subset_results, f, indent=2, default=str)
    print(f"Scaling results saved to {scaling_path}")


def run_bandwidth_sweep(config_path: str):
    """Run experiment multiple times with different bandwidth limits for H2 validation."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    bw_settings = config.get("bandwidth_sweep_bytes_per_frame", [999999999])
    base_name = config["experiment_name"]
    all_bw_results = {}

    for bw in bw_settings:
        bw_float = float(bw)
        bw_label = "unlimited" if bw_float > 1e8 else f"{int(bw_float)}"
        print(f"\n{'#'*60}")
        print(f"Bandwidth sweep: {bw_label} bytes/frame")
        print(f"{'#'*60}")

        results = run_experiment(config_path, bandwidth_bytes_per_frame=bw_float)
        all_bw_results[bw_label] = results

    # Print bandwidth sweep summary
    print(f"\n{'='*60}")
    print("H2 Bandwidth Sweep Summary")
    print(f"{'='*60}")
    print(f"{'BW (B/frame)':<20} {'Mean ATE (m)':<15} {'Bytes Sent':<15} {'Msgs Dropped':<15}")
    print("-" * 65)

    summary_rows = []
    for bw_label, res in all_bw_results.items():
        ates = []
        bytes_total = 0
        dropped = 0
        for scene, sr in res.items():
            for k, v in sr.items():
                if "ATE_RMSE" in k and isinstance(v, float):
                    ates.append(v)
            bytes_total += sr.get("comm_bytes_total", 0)
            dropped += sr.get("comm_dropped", 0)
        mean_ate = float(np.nanmean(ates)) if ates else float('nan')
        print(f"{bw_label:<20} {mean_ate:<15.4f} {bytes_total:<15} {dropped:<15}")
        summary_rows.append({
            "bandwidth_bytes_per_frame": bw_label,
            "mean_ATE_RMSE": mean_ate,
            "comm_bytes_total": bytes_total,
            "comm_dropped": dropped,
        })

    # Validate H2: graceful degradation
    ate_values = [r["mean_ATE_RMSE"] for r in summary_rows if not np.isnan(r["mean_ATE_RMSE"])]
    if len(ate_values) >= 2:
        ate_unlimited = ate_values[0]  # first = highest bandwidth
        ate_most_constrained = ate_values[-1]  # last = lowest bandwidth
        ratio = ate_most_constrained / ate_unlimited if ate_unlimited > 0 else float('inf')
        graceful = ratio < 5.0  # ATE shouldn't explode by more than 5x
        print(f"\nH2 Validation:")
        print(f"  ATE at unlimited BW: {ate_unlimited:.4f}m")
        print(f"  ATE at most constrained: {ate_most_constrained:.4f}m")
        print(f"  Degradation ratio: {ratio:.2f}x")
        print(f"  Graceful degradation (< 5x): {'PASS' if graceful else 'FAIL'}")

    # Save sweep results
    output_dir = config.get("output", {}).get("dir", "outputs")
    sweep_path = os.path.join(output_dir, "results", f"{base_name}_bw_sweep.json")
    os.makedirs(os.path.dirname(sweep_path), exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump({"sweep_summary": summary_rows, "all_results": all_bw_results},
                  f, indent=2, default=str)
    print(f"Sweep results saved to {sweep_path}")


def run_comm_sweep(config_path: str):
    """Run experiment multiple times with different loss rates for H2 validation.

    Tests: Does DUAG-C degrade gracefully when inter-robot messages are
    randomly dropped?  Loss rates from 0% (ideal) to 90% (near-isolation).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loss_rates = config.get("comm_sweep_loss_rates", [0.0])
    base_name = config["experiment_name"]
    all_results = {}

    for lr in loss_rates:
        lr_label = f"loss_{int(lr * 100)}pct"
        print(f"\n{'#'*60}")
        print(f"Comm sweep: loss_rate = {lr:.0%}")
        print(f"{'#'*60}")

        results = run_experiment(config_path, loss_rate=float(lr))
        all_results[lr_label] = results

    # Print sweep summary
    print(f"\n{'='*60}")
    print("H2 Communication Constraint Sweep Summary")
    print(f"{'='*60}")
    print(f"{'Loss Rate':<15} {'Mean ATE (m)':<15} {'Msgs Sent':<12} {'Msgs Dropped':<15}")
    print("-" * 57)

    summary_rows = []
    for lr_label, res in all_results.items():
        ates = []
        sent = 0
        dropped = 0
        for scene, sr in res.items():
            for k, v in sr.items():
                if "ATE_RMSE" in k and isinstance(v, float):
                    ates.append(v)
            sent += sr.get("comm_sent", 0)
            dropped += sr.get("comm_dropped", 0)
        mean_ate = float(np.nanmean(ates)) if ates else float('nan')
        print(f"{lr_label:<15} {mean_ate:<15.4f} {sent:<12} {dropped:<15}")
        summary_rows.append({
            "loss_rate": lr_label,
            "mean_ATE_RMSE": mean_ate,
            "comm_sent": sent,
            "comm_dropped": dropped,
        })

    # Validate H2: graceful degradation
    ate_values = [r["mean_ATE_RMSE"] for r in summary_rows if not np.isnan(r["mean_ATE_RMSE"])]
    if len(ate_values) >= 2:
        ate_ideal = ate_values[0]      # first = no loss
        ate_worst = ate_values[-1]     # last = highest loss
        ratio = ate_worst / ate_ideal if ate_ideal > 0 else float('inf')
        graceful = ratio < 5.0
        print(f"\nH2 Validation:")
        print(f"  ATE at 0% loss: {ate_ideal:.4f}m (target: < 0.20m)")
        print(f"  ATE at {int(float(loss_rates[-1]) * 100)}% loss: {ate_worst:.4f}m")
        print(f"  Degradation ratio: {ratio:.2f}x")
        print(f"  Graceful degradation (< 5x): {'PASS' if graceful else 'FAIL'}")

    # Save sweep results
    output_dir = config.get("output", {}).get("dir", "outputs")
    sweep_path = os.path.join(output_dir, "results", f"{base_name}_comm_sweep.json")
    os.makedirs(os.path.dirname(sweep_path), exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump({"sweep_summary": summary_rows, "all_results": all_results},
                  f, indent=2, default=str)
    print(f"Sweep results saved to {sweep_path}")


def main():
    parser = argparse.ArgumentParser(description="Run DUAG-C experiment")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # If config has comm_sweep_loss_rates, run the communication sweep
    if "comm_sweep_loss_rates" in config:
        run_comm_sweep(args.config)
    # If config has bandwidth_sweep, run the bandwidth sweep experiment
    elif "bandwidth_sweep_bytes_per_frame" in config:
        run_bandwidth_sweep(args.config)
    # If config has robot_subsets, run the N-robot scaling experiment
    elif "robot_subsets" in config.get("dataset", {}):
        run_kimera_subsets(args.config)
    else:
        run_experiment(args.config)


if __name__ == "__main__":
    main()
