#!/usr/bin/env python3
"""Step through init_map to find exact segfault location."""
import faulthandler; faulthandler.enable()
import sys, os, traceback
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'repos', 'MAGiC-SLAM'))
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.utils.io_utils import load_config
from src.utils.utils import setup_seed
from src.entities.agent import Agent

config_path = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/outputs/baselines/magic_slam/apart_0_single_gpu.yaml'

config = load_config(config_path)
setup_seed(config["seed"])
agent = Agent(0, {}, config)
print("Agent created", flush=True)

# Step through init_map manually
from src.entities.gaussian_model import GaussianModel
from src.utils.utils import torch2np, np2ptcloud, get_render_settings
import numpy as np
import torch

print("Step 1: GaussianModel(0)", flush=True)
gm = GaussianModel(0)
gm.training_setup(agent.opt)
print(f"Step 2: GaussianModel created, size={gm.get_size()}", flush=True)

# Call mapper.map manually step by step
frame_id = 0
estimate_c2w = torch2np(agent.gt_c2ws[0])
estimate_w2c = np.linalg.inv(estimate_c2w)

print("Step 3: getting dataset[0]", flush=True)
_, gt_color, gt_depth, _ = agent.dataset[0]
print(f"  gt_color: {type(gt_color)} shape={gt_color.shape} dtype={gt_color.dtype}", flush=True)
print(f"  gt_depth: {type(gt_depth)} shape={gt_depth.shape} dtype={gt_depth.dtype}", flush=True)

print("Step 4: color_transform", flush=True)
import torchvision
color_transform = torchvision.transforms.ToTensor()
color_tensor = color_transform(gt_color).cuda()
print(f"  color_tensor: {color_tensor.shape}", flush=True)

print("Step 5: depth tensor", flush=True)
from src.utils.utils import np2torch
depth_tensor = np2torch(gt_depth, device="cuda")
print(f"  depth_tensor: {depth_tensor.shape}", flush=True)

print("Step 6: render_settings", flush=True)
render_settings = get_render_settings(
    agent.dataset.width, agent.dataset.height, agent.dataset.intrinsics, estimate_w2c)
print("  render_settings OK", flush=True)

keyframe = {"color": color_tensor, "depth": depth_tensor, "render_settings": render_settings}

print("Step 7: compute_seeding_mask", flush=True)
seeding_mask = agent.mapper.compute_seeding_mask(gm, keyframe, True)
print(f"  seeding_mask: type={type(seeding_mask)}", flush=True)

print("Step 8: seed_new_gaussians", flush=True)
pts = agent.mapper.seed_new_gaussians(
    gt_color, gt_depth, agent.dataset.intrinsics, estimate_c2w, seeding_mask, True)
print(f"  pts: shape={pts.shape} dtype={pts.dtype}", flush=True)
print(f"  pts min={pts.min():.4f} max={pts.max():.4f}", flush=True)
print(f"  pts[:3] = {pts[:3]}", flush=True)
print(f"  pts NaN count: {np.isnan(pts).sum()}", flush=True)
print(f"  pts Inf count: {np.isinf(pts).sum()}", flush=True)

print("Step 9: np2ptcloud", flush=True)
pts_xyz = pts[:, :3]
pts_rgb = pts[:, 3:] / 255.0
print(f"  pts_xyz: shape={pts_xyz.shape} dtype={pts_xyz.dtype} contig={pts_xyz.flags['C_CONTIGUOUS']}", flush=True)
print(f"  pts_rgb: shape={pts_rgb.shape} dtype={pts_rgb.dtype} contig={pts_rgb.flags['C_CONTIGUOUS']}", flush=True)

import open3d as o3d
print("  calling Vector3dVector for points...", flush=True)
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(pts_xyz.astype(np.float64))
print(f"  points assigned: {len(cloud.points)}", flush=True)
cloud.colors = o3d.utility.Vector3dVector(pts_rgb.astype(np.float64))
print(f"  colors assigned: {len(cloud.colors)}", flush=True)

print("Step 10: add_points", flush=True)
gm.add_points(cloud)
print(f"  GaussianModel size after add: {gm.get_size()}", flush=True)

print("ALL STEPS PASSED!", flush=True)
