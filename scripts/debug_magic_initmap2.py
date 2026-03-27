#!/usr/bin/env python3
"""Call agent.init_map() exactly as run() does."""
import faulthandler; faulthandler.enable()
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'repos', 'MAGiC-SLAM'))
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.utils.io_utils import load_config
from src.utils.utils import setup_seed
from src.entities.agent import Agent

config = load_config(sys.argv[1] if len(sys.argv) > 1 else
    '/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/outputs/baselines/magic_slam/apart_0_single_gpu.yaml')
setup_seed(config["seed"])
agent = Agent(0, {}, config)
print("Agent created", flush=True)

# Exactly what run() does before init_map
setup_seed(config["seed"])  # run() calls this again

print("Calling agent.init_map()...", flush=True)
sys.stdout.flush()
gm = agent.init_map()
print(f"init_map() OK, model size={gm.get_size()}", flush=True)

# Continue: extract features for frame 0
from PIL import Image
image = Image.fromarray(agent.dataset[0][1])
agent.submap_features[agent.submap_id] = agent.feature_extractor.extract_features(image).cpu().numpy()
print("Features extracted OK", flush=True)

# Process some frames
for frame_id in range(1, 10):
    print(f"Frame {frame_id}...", end=" ", flush=True)
    if frame_id == 1:
        estimated_c2w = agent.dataset[frame_id][-1]
    else:
        from src.utils.utils import torch2np
        import torch
        estimated_c2w = agent.tracker.track(
            frame_id, gm,
            torch2np(agent.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
    from src.utils.utils import np2torch
    agent.estimated_c2ws[frame_id] = np2torch(estimated_c2w)
    gm.training_setup(agent.opt)
    start_new_submap = agent.should_start_new_submap(frame_id)
    if start_new_submap:
        print("NEW SUBMAP", flush=True)
    elif agent.should_start_mapping(frame_id):
        agent.keyframe_ids.append(frame_id)
        agent.mapper.map(frame_id, estimated_c2w, gm, False, agent.mapper.iterations)
        print(f"mapped (size={gm.get_size()})", flush=True)
    else:
        print("skip", flush=True)

print("ALL OK!", flush=True)
