#!/usr/bin/env python3
"""
Run MAGiC-SLAM 2-agent for a single scene with monkey-patched max_threads=2.
Prevents the buffer overflow crash that occurs with max_threads=20.

Usage:
  /home/wen/anaconda3/envs/magic-slam/bin/python scripts/run_magic_single_scene.py <config_path>

Example:
  /home/wen/anaconda3/envs/magic-slam/bin/python scripts/run_magic_single_scene.py \
      outputs/baselines/magic_slam/apart_1_2agent_single_gpu.yaml
"""
import sys, os

# Must run from MAGiC-SLAM directory
WORKSPACE = "/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM"
MAGIC_DIR = os.path.join(WORKSPACE, "repos/MAGiC-SLAM")
os.chdir(MAGIC_DIR)
sys.path.insert(0, MAGIC_DIR)

# Monkey-patch optimize_poses to use max_threads=2
import src.entities.magic_slam as ms

_orig_optimize = ms.MAGiCSLAM.optimize_poses

def _patched_optimize(self, agents_submaps):
    import numpy as np
    from src.entities.loop_detection.loop_detector import LoopDetector
    from src.entities.pose_graph_adapter import PoseGraphAdapter
    from src.utils.magic_slam_utils import (
        register_agents_submaps, register_submaps, apply_pose_correction
    )

    loop_detector = LoopDetector(self.config["loop_detection"])
    agents_c2ws = {}
    for agent_id, agent_submaps in agents_submaps.items():
        agents_c2ws[agent_id] = np.vstack(
            [s["submap_c2ws"] for s in agent_submaps])

    intra_loops, inter_loops = loop_detector.detect_loops(agents_submaps)
    print(f"[PATCHED] Detected {len(intra_loops)} intra + {len(inter_loops)} inter loops")

    # KEY FIX: max_threads=2 instead of 20 to prevent buffer overflow
    print("[PATCHED] Running ICP with max_threads=2...")
    intra_loops = register_agents_submaps(
        agents_submaps, intra_loops, register_submaps, max_threads=2)
    inter_loops = register_agents_submaps(
        agents_submaps, inter_loops, register_submaps, max_threads=2)

    intra_loops = loop_detector.filter_loops(intra_loops)
    inter_loops = loop_detector.filter_loops(inter_loops)
    print(f"[PATCHED] After filtering: {len(intra_loops)} intra + {len(inter_loops)} inter loops")

    graph_wrapper = PoseGraphAdapter(agents_submaps, intra_loops + inter_loops)
    if len(intra_loops + inter_loops) > 0:
        print("[PATCHED] Running PGO...")
        graph_wrapper.optimize()

    return apply_pose_correction(graph_wrapper.get_poses(), agents_submaps)

ms.MAGiCSLAM.optimize_poses = _patched_optimize
print("[PATCHED] optimize_poses monkey-patched with max_threads=2")


def main():
    # Now run the normal pipeline
    from run_slam import get_args, update_config_with_args
    from src.entities.magic_slam import MAGiCSLAM
    from src.utils.io_utils import load_config
    from src.utils.utils import setup_seed

    # Override sys.argv for argparse
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not config_path:
        print("Usage: python scripts/run_magic_single_scene.py <config_path>")
        sys.exit(1)

    # Handle relative paths from workspace
    if not os.path.isabs(config_path):
        config_path = os.path.join(WORKSPACE, config_path)

    sys.argv = ["run_slam.py", config_path]
    args = get_args()
    config = load_config(args.config_path)
    config = update_config_with_args(config, args)

    print(f"Scene: {config.get('data', {}).get('scene_name', '?')}")
    print(f"Output: {config.get('data', {}).get('output_path', '?')}")
    print(f"Agents: {config.get('data', {}).get('agent_ids', '?')}")
    print(f"Multi-GPU: {config.get('multi_gpu', '?')}")

    setup_seed(config["seed"])
    slam = MAGiCSLAM(config)
    slam.run()
    print("\n[DONE] Scene completed successfully!")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()
