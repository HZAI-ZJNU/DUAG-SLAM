#!/usr/bin/env python3
"""Run a single MAGiC-SLAM agent in the main process to capture errors."""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'repos', 'MAGiC-SLAM'))

from src.utils.io_utils import load_config
from src.utils.utils import setup_seed
from src.entities.agent import Agent
from multiprocessing import Pipe

config_path = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/outputs/baselines/magic_slam/apart_0_single_gpu.yaml'

# MAGiC-SLAM uses relative config paths (inherit_from), so we must cd to its dir
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'repos', 'MAGiC-SLAM'))
print(f"CWD: {os.getcwd()}")
print("Loading config...")
config = load_config(config_path)
setup_seed(config["seed"])

print("Creating agent 0...")
try:
    # Check numpy/cv2 BEFORE creating agent
    import numpy as np
    import cv2
    print(f"numpy: {np.__version__} from {np.__file__}")
    print(f"cv2: {cv2.__version__} from {cv2.__file__}")
    test_img = cv2.imread('/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/data/ReplicaMultiagent/Apart-0/apart_0_part1/results/frame000000.jpg')
    print(f"cv2.imread isinstance: {isinstance(test_img, np.ndarray)}")
    print(f"id(type(test_img)): {id(type(test_img))}")
    print(f"id(np.ndarray): {id(np.ndarray)}")

    agent = Agent(0, {}, config)
    print("Agent created successfully")

    # Check dataset type
    _, gt_color, _, _ = agent.dataset[0]
    print(f"gt_color type: {type(gt_color)}, isinstance: {isinstance(gt_color, np.ndarray)}")
    print(f"id(type(gt_color)): {id(type(gt_color))}")

    # Try to_tensor manually
    import torchvision
    t = torchvision.transforms.ToTensor()(gt_color)
    print(f"ToTensor OK: {t.shape}")

    parent_conn, child_conn = Pipe()
    agent.set_pipe(child_conn)
    print("Starting agent.run() in main process...")
    agent.run()
    print("Agent finished!")
except Exception:
    traceback.print_exc()
    sys.exit(1)
