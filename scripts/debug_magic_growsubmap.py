#!/usr/bin/env python3
"""Call grow_submap exactly as mapper.map() does."""
import faulthandler; faulthandler.enable()
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'repos', 'MAGiC-SLAM'))
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.utils.io_utils import load_config
from src.utils.utils import setup_seed, torch2np, np2torch, get_render_settings
from src.entities.agent import Agent
from src.entities.gaussian_model import GaussianModel
import torchvision, numpy as np, torch

config = load_config(sys.argv[1] if len(sys.argv) > 1 else
    '/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/outputs/baselines/magic_slam/apart_0_single_gpu.yaml')
setup_seed(config["seed"])
agent = Agent(0, {}, config)
print("Agent created", flush=True)

gm = GaussianModel(0)
gm.training_setup(agent.opt)

# Replicate map() exactly
frame_id = 0
estimate_c2w = torch2np(agent.gt_c2ws[0])
estimate_w2c = np.linalg.inv(estimate_c2w)
_, gt_color, gt_depth, _ = agent.dataset[0]

color_transform = torchvision.transforms.ToTensor()
keyframe = {
    "color": color_transform(gt_color).cuda(),
    "depth": np2torch(gt_depth, device="cuda"),
    "render_settings": get_render_settings(
        agent.dataset.width, agent.dataset.height, agent.dataset.intrinsics, estimate_w2c)}
print("Keyframe prepared", flush=True)

seeding_mask = agent.mapper.compute_seeding_mask(gm, keyframe, True)
print(f"Seeding mask: {seeding_mask.shape}, sum={seeding_mask.sum()}", flush=True)

pts = agent.mapper.seed_new_gaussians(
    gt_color, gt_depth, agent.dataset.intrinsics, estimate_c2w, seeding_mask, True)
print(f"pts: {pts.shape} {pts.dtype}", flush=True)

# Now call grow_submap EXACTLY as mapper does
print("Calling grow_submap...", flush=True)
sys.stdout.flush()
new_pts = agent.mapper.grow_submap(pts, gm)
print(f"grow_submap OK: {new_pts} points added, model size={gm.get_size()}", flush=True)

# Now try optimize_submap (1000 iterations like init_map)
print("Calling optimize_submap (1000 iters)...", flush=True)
sys.stdout.flush()
opt_dict = agent.mapper.optimize_submap([(0, keyframe)], gm, 1000)
print(f"optimize_submap OK: {opt_dict.get('optimization_time', 'N/A')}", flush=True)
print("DONE!", flush=True)
