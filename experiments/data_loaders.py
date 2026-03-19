# experiments/data_loaders.py
# Dataset-specific loaders. All return dict: key -> list of (rgb, depth, timestamp, gt_pose)
# All datasets converted to TUM format: groundtruth.txt = "timestamp tx ty tz qx qy qz qw"

import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def _load_tum_trajectory(traj_path: str) -> dict:
    """Returns {timestamp: torch.Tensor [4,4] SE(3)} from TUM groundtruth.txt."""
    from scipy.spatial.transform import Rotation
    poses = {}
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = torch.eye(4)
            T[:3, :3] = torch.from_numpy(R).float()
            T[:3, 3] = torch.tensor([tx, ty, tz])
            poses[ts] = T
    return poses


def _load_rgb(path: str) -> torch.Tensor:
    """Load RGB image as [H,W,3] float32 in [0,1]."""
    img = Image.open(path).convert('RGB')
    return torch.from_numpy(np.array(img)).float() / 255.0


def _load_depth(path: str, scale: float = 1000.0) -> torch.Tensor:
    """Load depth image as [H,W,1] float32 in meters."""
    d = np.array(Image.open(path)).astype(np.float32) / scale
    return torch.from_numpy(d).unsqueeze(-1)


def load_replica_multiagent(config: dict) -> dict:
    """
    Loader for Replica-Multiagent dataset (MAGiC-SLAM format).
    VERIFY actual folder structure with: ls data/replica_multiagent/
    Expected: data/replica_multiagent/{scene}/agent_{id}/rgb/, depth/, traj.txt
    """
    path = config['dataset']['path']
    scenes = config['dataset']['scenes']
    n_agents = config['dataset']['n_agents']
    dataset = {}

    for scene in scenes:
        for agent_id in range(n_agents):
            key = (scene, agent_id)
            agent_dir = os.path.join(path, scene, f"agent_{agent_id}")
            rgb_dir = os.path.join(agent_dir, "rgb")
            depth_dir = os.path.join(agent_dir, "depth")
            traj_path = os.path.join(agent_dir, "traj.txt")

            if not os.path.exists(agent_dir):
                raise FileNotFoundError(
                    f"Missing: {agent_dir}\n"
                    f"Verify folder structure with: ls {path}/{scene}/")

            gt_poses = _load_tum_trajectory(traj_path)
            rgb_files = sorted(Path(rgb_dir).glob("*.png"))
            depth_files = sorted(Path(depth_dir).glob("*.png"))

            frames = []
            for rgb_f, depth_f in zip(rgb_files, depth_files):
                ts = float(rgb_f.stem)
                rgb = _load_rgb(str(rgb_f))
                depth = _load_depth(str(depth_f), scale=6553.5)  # Replica depth scale
                gt = gt_poses.get(ts, torch.eye(4))
                frames.append((rgb, depth, ts, gt))

            dataset[key] = frames

    return dataset


def load_aria_multiagent(config: dict) -> dict:
    """
    Loader for Aria-Multiagent dataset.
    VERIFY actual folder structure: ls data/aria_multiagent/
    Expected: data/aria_multiagent/{scene}/agent_{id}/rgb/, depth/, groundtruth.txt
    """
    path = config['dataset']['path']
    scenes = config['dataset']['scenes']
    n_agents = config['dataset']['n_agents']
    dataset = {}

    for scene in scenes:
        for agent_id in range(n_agents):
            key = (scene, agent_id)
            agent_dir = os.path.join(path, scene, f"agent_{agent_id}")
            if not os.path.exists(agent_dir):
                raise FileNotFoundError(f"Missing: {agent_dir}. Verify download.")

            gt_poses = _load_tum_trajectory(os.path.join(agent_dir, "groundtruth.txt"))
            rgb_files = sorted(Path(agent_dir, "rgb").glob("*.jpg")) or \
                        sorted(Path(agent_dir, "rgb").glob("*.png"))
            depth_files = sorted(Path(agent_dir, "depth").glob("*.png"))

            frames = []
            for rgb_f, depth_f in zip(rgb_files, depth_files):
                ts = float(rgb_f.stem)
                rgb = _load_rgb(str(rgb_f))
                depth = _load_depth(str(depth_f), scale=1000.0)
                gt = gt_poses.get(ts, torch.eye(4))
                frames.append((rgb, depth, ts, gt))
            dataset[key] = frames
    return dataset


def load_s3e(config: dict) -> dict:
    """
    Loader for S3E dataset.
    VERIFY actual folder structure: ls data/s3e/
    Expected: data/s3e/{sequence}/robot_{id}/rgb/, depth/, groundtruth.txt
    """
    path = config['dataset']['path']
    paradigms = config['dataset'].get('paradigms', {})
    n_agents = config['dataset']['n_agents']
    dataset = {}
    all_seqs = [seq for seqs in paradigms.values() for seq in seqs] if paradigms else config['dataset'].get('scenes', [])

    for seq in all_seqs:
        for robot_id in range(n_agents):
            key = (seq, robot_id)
            robot_dir = os.path.join(path, seq, f"robot_{robot_id}")
            if not os.path.exists(robot_dir):
                raise FileNotFoundError(f"Missing: {robot_dir}. Check S3E download.")

            gt_poses = _load_tum_trajectory(os.path.join(robot_dir, "groundtruth.txt"))
            rgb_files = sorted(Path(robot_dir, "rgb").glob("*.png"))
            depth_files = sorted(Path(robot_dir, "depth").glob("*.png"))

            frames = []
            for rgb_f, depth_f in zip(rgb_files, depth_files):
                ts = float(rgb_f.stem)
                rgb = _load_rgb(str(rgb_f))
                depth = _load_depth(str(depth_f), scale=1000.0)
                gt = gt_poses.get(ts, torch.eye(4))
                frames.append((rgb, depth, ts, gt))
            dataset[key] = frames
    return dataset


def load_kimera_campus(config: dict) -> dict:
    """
    Loader for Kimera-Multi Campus-Outdoor dataset.
    VERIFY: ls data/kimera_campus/
    """
    path = config['dataset']['path']
    sequences = config['dataset'].get('sequences', config['dataset'].get('scenes', []))
    robot_subsets = config['dataset'].get('robot_subsets', [[0, 1]])
    n_agents = max(max(subset) for subset in robot_subsets) + 1
    dataset = {}

    for seq in sequences:
        for robot_id in range(n_agents):
            key = (seq, robot_id)
            robot_dir = os.path.join(path, seq, f"robot_{robot_id}")
            if not os.path.exists(robot_dir):
                continue  # some subsets not used

            gt_poses = _load_tum_trajectory(os.path.join(robot_dir, "groundtruth.txt"))
            rgb_files = sorted(Path(robot_dir, "rgb").glob("*.png"))
            depth_files = sorted(Path(robot_dir, "depth").glob("*.png"))

            frames = []
            for rgb_f, depth_f in zip(rgb_files, depth_files):
                ts = float(rgb_f.stem)
                rgb = _load_rgb(str(rgb_f))
                depth = _load_depth(str(depth_f), scale=1000.0)
                gt = gt_poses.get(ts, torch.eye(4))
                frames.append((rgb, depth, ts, gt))
            dataset[key] = frames
    return dataset


def load_subtmrs(config: dict) -> dict:
    """
    Loader for SubT-MRS dataset.
    VERIFY: ls data/subtmrs/
    """
    path = config['dataset']['path']
    seqs = config['dataset'].get('sequences', config['dataset'].get('scenes', []))
    n_agents = config['dataset']['n_agents']
    dataset = {}

    for seq in seqs:
        for robot_id in range(n_agents):
            key = (seq, robot_id)
            robot_dir = os.path.join(path, seq, f"robot_{robot_id}")
            if not os.path.exists(robot_dir):
                continue

            gt_poses = _load_tum_trajectory(os.path.join(robot_dir, "groundtruth.txt"))
            rgb_files = sorted(Path(robot_dir, "rgb").glob("*.png"))
            depth_files = sorted(Path(robot_dir, "depth").glob("*.png"))

            frames = []
            for rgb_f, depth_f in zip(rgb_files, depth_files):
                ts = float(rgb_f.stem)
                rgb = _load_rgb(str(rgb_f))
                depth = _load_depth(str(depth_f), scale=1000.0)
                gt = gt_poses.get(ts, torch.eye(4))
                frames.append((rgb, depth, ts, gt))
            dataset[key] = frames
    return dataset


LOADERS = {
    "replica_multiagent": load_replica_multiagent,
    "aria_multiagent":    load_aria_multiagent,
    "s3e":                load_s3e,
    "kimera_campus":      load_kimera_campus,
    "subtmrs":            load_subtmrs,
}


def load_dataset(config: dict) -> dict:
    """Main entry point. Routes to correct loader based on config['dataset']['name']."""
    name = config['dataset']['name']
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS.keys())}")
    return LOADERS[name](config)
