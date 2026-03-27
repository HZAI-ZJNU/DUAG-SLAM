#!/usr/bin/env python3
"""Run a single MAGiC-SLAM agent in main process with detailed error capture."""
import faulthandler
faulthandler.enable()
import sys, os, traceback
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'repos', 'MAGiC-SLAM'))
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from multiprocessing import Pipe
from src.utils.io_utils import load_config
from src.utils.utils import setup_seed
from src.entities.agent import Agent

config_path = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/outputs/baselines/magic_slam/apart_0_single_gpu.yaml'

print(f"CWD: {os.getcwd()}", flush=True)
config = load_config(config_path)
setup_seed(config["seed"])

print("Creating agent 0...", flush=True)
try:
    agent = Agent(0, {}, config)
    print("Agent created OK", flush=True)
    
    parent_conn, child_conn = Pipe()
    agent.set_pipe(child_conn)
    
    print("Calling agent.run()...", flush=True)
    agent.run()
    print("agent.run() completed!", flush=True)
    
    # Read result from pipe
    print("Agent finished, checking pipe...", flush=True)
    
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
