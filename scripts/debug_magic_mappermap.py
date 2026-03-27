#!/usr/bin/env python3
"""Call mapper.map() exactly as init_map does."""
import faulthandler; faulthandler.enable()
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'repos', 'MAGiC-SLAM'))
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.utils.io_utils import load_config
from src.utils.utils import setup_seed, torch2np
from src.entities.agent import Agent
from src.entities.gaussian_model import GaussianModel

config = load_config(sys.argv[1] if len(sys.argv) > 1 else
    '/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/outputs/baselines/magic_slam/apart_0_single_gpu.yaml')
setup_seed(config["seed"])
agent = Agent(0, {}, config)
print("Agent created", flush=True)

# Exactly as init_map does
gm = GaussianModel(0)
gm.training_setup(agent.opt)
print("Calling mapper.map(0, ..., True, 1000)...", flush=True)
sys.stdout.flush()
agent.mapper.map(0, torch2np(agent.gt_c2ws[0]), gm, True, 1000)
print(f"mapper.map() DONE, model size={gm.get_size()}", flush=True)

# Now continue with what run() does after init_map
from PIL import Image
image = Image.fromarray(agent.dataset[0][1])
print("Image created for feature extraction", flush=True)
feats = agent.feature_extractor.extract_features(image).cpu().numpy()
print(f"Features extracted: {feats.shape}", flush=True)

# Frame 1
print("Processing frame 1...", flush=True)
gm.training_setup(agent.opt)
estimated_c2w = agent.dataset[1][-1]  # GT pose for frame 1
agent.mapper.map(1, estimated_c2w, gm, False, agent.mapper.iterations)
print("Frame 1 mapped!", flush=True)

print("SUCCESS - no segfault", flush=True)
