#!/usr/bin/env python3
"""
MAGiC-SLAM Recovery Script — loads saved submaps + poses, runs PGO + merge + 
refine + eval. Avoids re-running the 1-hour tracking phase.

Can also be used for fresh 2-agent runs with reduced ICP threads to prevent
the buffer overflow crash.

Usage:
  # Recovery mode (use saved submaps, optionally skip or run PGO):
  python scripts/magic_recover_and_eval.py --scene Apart-0 --mode recover --do_pgo

  # Recovery mode, skip PGO (fastest):
  python scripts/magic_recover_and_eval.py --scene Apart-0 --mode recover --no_pgo

  # All scenes:
  python scripts/magic_recover_and_eval.py --scene all --mode recover --do_pgo

Must be run with: /home/wen/anaconda3/envs/magic-slam/bin/python
from the repos/MAGiC-SLAM/ directory (or set cwd).
"""

import sys, os, json, time
import argparse
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser as AP

# ── Setup paths ──────────────────────────────────────────────────────────────
WORKSPACE = "/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM"
MAGIC_DIR = os.path.join(WORKSPACE, "repos/MAGiC-SLAM")
os.chdir(MAGIC_DIR)
sys.path.insert(0, MAGIC_DIR)

from src.entities.gaussian_model import GaussianModel
from src.entities.datasets import Replica
from src.entities.arguments import OptimizationParams
from src.utils.magic_slam_utils import (
    merge_submaps, refine_map, register_agents_submaps,
    register_submaps, apply_pose_correction
)
from src.utils.mapping_eval import eval_agents_rendering
from src.utils.tracking_eval import compute_ate, align_trajectories


# ── Scene configs ────────────────────────────────────────────────────────────
SCENES = {
    "Apart-0": {"frame_limit": 2500},
    "Apart-1": {"frame_limit": 2500},
    "Apart-2": {"frame_limit": 2500},
    "Office-0": {"frame_limit": 1500},
}

CAM_CONFIG = {
    "H": 680, "W": 1200,
    "fx": 600.0, "fy": 600.0,
    "cx": 599.5, "cy": 339.5,
    "depth_scale": 6553.5,
}

LOOP_DETECTION_CONFIG = {
    "feature_extractor_name": "dino",
    "weights_path": os.path.join(WORKSPACE, "models/dinov2-small"),
    "embed_size": 384,
    "feature_dist_threshold": 0.35,
    "device": "cpu",
    "time_threshold": 0,
    "max_loops_per_frame": 1,
    "fitness_threshold": 0.5,
    "inlier_rmse_threshold": 0.1,
}


def get_opt_args():
    """Create OptimizationParams with default values."""
    return OptimizationParams(AP(description="opt"))


def load_submaps(agent_dir: Path):
    """Load all submap checkpoints from an agent directory."""
    submap_dir = agent_dir / "submaps"
    submap_files = sorted(submap_dir.glob("*.ckpt"))
    submaps = []
    for sf in submap_files:
        sm = torch.load(sf, map_location="cpu", weights_only=False)
        submaps.append(sm)
    print(f"  Loaded {len(submaps)} submaps from {agent_dir.name}")
    return submaps


def build_kf_ids(submaps):
    """Extract keyframe IDs from submaps."""
    kf_ids = np.empty((0,))
    for sm in submaps:
        kf_ids = np.concatenate([kf_ids, sm["keyframe_ids"]])
    return kf_ids.astype(int)


def build_kf_c2ws_no_pgo(submaps, kf_ids):
    """Build keyframe c2ws WITHOUT PGO correction (identity delta per submap)."""
    kf_c2ws = []
    for sm in submaps:
        sm_kf_ids = sm["keyframe_ids"] - sm["submap_start_frame_id"]
        sm_kf_ids = sm_kf_ids.astype(int)
        for kid in sm_kf_ids:
            kf_c2ws.append(sm["submap_c2ws"][kid])
    return np.array(kf_c2ws)


def build_gt_kf_c2ws(submaps, kf_ids):
    """Build ground truth keyframe c2ws from submaps."""
    gt_c2ws = []
    for sm in submaps:
        sm_kf_ids = sm["keyframe_ids"] - sm["submap_start_frame_id"]
        sm_kf_ids = sm_kf_ids.astype(int)
        for kid in sm_kf_ids:
            gt_c2ws.append(sm["submap_gt_c2ws"][kid])
    return np.array(gt_c2ws)


def run_pgo(agents_submaps, max_threads=2):
    """Run loop detection + ICP registration + PGO with reduced threads."""
    from src.entities.loop_detection.loop_detector import LoopDetector
    from src.entities.pose_graph_adapter import PoseGraphAdapter

    print("  Running loop detection...")
    loop_detector = LoopDetector(LOOP_DETECTION_CONFIG)
    intra_loops, inter_loops = loop_detector.detect_loops(agents_submaps)
    print(f"  Detected {len(intra_loops)} intra-agent + {len(inter_loops)} inter-agent loops")

    print(f"  Running ICP registration (max_threads={max_threads})...")
    t0 = time.time()
    intra_loops = register_agents_submaps(
        agents_submaps, intra_loops, register_submaps, max_threads=max_threads)
    inter_loops = register_agents_submaps(
        agents_submaps, inter_loops, register_submaps, max_threads=max_threads)
    print(f"  ICP done in {time.time()-t0:.1f}s")

    intra_loops = loop_detector.filter_loops(intra_loops)
    inter_loops = loop_detector.filter_loops(inter_loops)
    print(f"  After filtering: {len(intra_loops)} intra + {len(inter_loops)} inter loops")

    all_loops = intra_loops + inter_loops
    graph_wrapper = PoseGraphAdapter(agents_submaps, all_loops)
    if len(all_loops) > 0:
        print("  Running PGO...")
        graph_wrapper.optimize()

    agents_opt_kf_c2ws = apply_pose_correction(
        graph_wrapper.get_poses(), agents_submaps)
    return agents_opt_kf_c2ws


def create_dataset(scene_name, agent_id, frame_limit):
    """Create a Replica dataset for an agent."""
    agent_paths = sorted(Path(os.path.join(
        WORKSPACE, "data/ReplicaMultiagent", scene_name)).glob("*"))
    agent_path = str(agent_paths[agent_id])
    config = {
        **CAM_CONFIG,
        "input_path": agent_path,
        "frame_limit": frame_limit,
    }
    return Replica(config)


def process_scene(scene_name, do_pgo=True, max_threads=2):
    """Process one scene: load saved data, optionally PGO, merge, refine, eval."""
    print(f"\n{'='*60}")
    print(f"Processing: {scene_name}")
    print(f"{'='*60}")

    frame_limit = SCENES[scene_name]["frame_limit"]
    output_dir = Path(WORKSPACE) / "outputs/baselines/magic_slam" / f"{scene_name}-2agent"
    agent_ids = [0, 1]

    # Check if submaps exist
    for aid in agent_ids:
        agent_dir = output_dir / f"agent_{aid}" / "submaps"
        if not agent_dir.exists() or len(list(agent_dir.glob("*.ckpt"))) == 0:
            print(f"  ERROR: No submaps found at {agent_dir}")
            print(f"  Need to run full pipeline for this scene.")
            return None

    # Load submaps
    print("\n[1/6] Loading submaps...")
    agents_submaps = {}
    for aid in agent_ids:
        agent_dir = output_dir / f"agent_{aid}"
        agents_submaps[aid] = load_submaps(agent_dir)

    # Build kf_ids
    print("\n[2/6] Building keyframe IDs...")
    agents_kf_ids = {}
    for aid in agent_ids:
        agents_kf_ids[aid] = build_kf_ids(agents_submaps[aid])
        print(f"  Agent {aid}: {len(agents_kf_ids[aid])} keyframes")

    # PGO or skip
    if do_pgo:
        print(f"\n[3/6] Running PGO (max_threads={max_threads})...")
        agents_opt_kf_c2ws = run_pgo(agents_submaps, max_threads=max_threads)
    else:
        print("\n[3/6] Skipping PGO — using raw estimated poses...")
        agents_opt_kf_c2ws = {}
        for aid in agent_ids:
            agents_opt_kf_c2ws[aid] = build_kf_c2ws_no_pgo(
                agents_submaps[aid], agents_kf_ids[aid])

    # ATE computation
    print("\n  Computing ATE...")
    results = {"scene": scene_name, "pgo": do_pgo}
    for aid in agent_ids:
        gt_c2ws = build_gt_kf_c2ws(agents_submaps[aid], agents_kf_ids[aid])
        est_c2ws = agents_opt_kf_c2ws[aid]

        gt_trans = gt_c2ws[:, :3, 3]
        est_trans = est_c2ws[:, :3, 3]

        n = min(len(gt_trans), len(est_trans))
        gt_trans = gt_trans[:n]
        est_trans = est_trans[:n]

        # Unaligned ATE
        ate_unaligned = compute_ate(est_trans, gt_trans)
        print(f"  Agent {aid} ATE RMSE (unaligned): {ate_unaligned['rmse']*100:.2f} cm")

        # Aligned ATE
        est_aligned = align_trajectories(est_trans, gt_trans)
        ate_aligned = compute_ate(est_aligned, gt_trans)
        print(f"  Agent {aid} ATE RMSE (aligned):   {ate_aligned['rmse']*100:.2f} cm")

        results[f"agent_{aid}_ate_rmse_cm"] = ate_unaligned['rmse'] * 100
        results[f"agent_{aid}_ate_aligned_cm"] = ate_aligned['rmse'] * 100

    # Create datasets for rendering eval
    print("\n[4/6] Creating datasets for rendering evaluation...")
    agents_datasets = {}
    for aid in agent_ids:
        agents_datasets[aid] = create_dataset(scene_name, aid, frame_limit)
        print(f"  Agent {aid}: {len(agents_datasets[aid])} frames")

    # Merge submaps
    print("\n[5/6] Merging submaps...")
    opt_args = get_opt_args()
    merged_map = merge_submaps(agents_submaps, agents_kf_ids, agents_opt_kf_c2ws, opt_args)
    merged_ply = str(output_dir / "merged_coarse.ply")
    merged_map.save_ply(merged_ply)
    print(f"  Saved coarse map: {merged_ply}")
    print(f"  Total Gaussians: {merged_map.get_size()}")

    # Eval coarse
    print("\n  Evaluating coarse map...")
    try:
        c_psnr, c_lpips, c_ssim, c_dl1 = eval_agents_rendering(
            merged_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws)
        results["coarse_psnr"] = c_psnr
        results["coarse_lpips"] = c_lpips
        results["coarse_ssim"] = c_ssim
        results["coarse_depth_l1"] = c_dl1
        print(f"  Coarse — PSNR: {c_psnr:.2f}, LPIPS: {c_lpips:.3f}, SSIM: {c_ssim:.3f}, DepthL1: {c_dl1:.3f}")
    except Exception as e:
        print(f"  Coarse eval failed: {e}")

    # Refine
    print("\n[6/6] Refining map (3000 iterations)...")
    t0 = time.time()
    refined_map = refine_map(
        merged_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws,
        iterations=3000)
    print(f"  Refinement done in {time.time()-t0:.1f}s")
    refined_ply = str(output_dir / "merged_refined.ply")
    refined_map.save_ply(refined_ply)
    print(f"  Saved refined map: {refined_ply}")

    # Eval refined
    print("\n  Evaluating refined map...")
    try:
        f_psnr, f_lpips, f_ssim, f_dl1 = eval_agents_rendering(
            refined_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws)
        results["fine_psnr"] = f_psnr
        results["fine_lpips"] = f_lpips
        results["fine_ssim"] = f_ssim
        results["fine_depth_l1"] = f_dl1
        print(f"  Refined — PSNR: {f_psnr:.2f}, LPIPS: {f_lpips:.3f}, SSIM: {f_ssim:.3f}, DepthL1: {f_dl1:.3f}")
    except Exception as e:
        print(f"  Refined eval failed: {e}")

    # Save results
    results_path = output_dir / "recovery_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="all",
                        help="Scene name (Apart-0, Apart-1, etc.) or 'all'")
    parser.add_argument("--do_pgo", action="store_true", default=True,
                        help="Run PGO (loop detection + ICP + graph optimization)")
    parser.add_argument("--no_pgo", action="store_true",
                        help="Skip PGO, use raw poses")
    parser.add_argument("--max_threads", type=int, default=2,
                        help="Max threads for ICP registration (default: 2, crash was at 20)")
    args = parser.parse_args()

    if args.no_pgo:
        args.do_pgo = False

    scenes = list(SCENES.keys()) if args.scene == "all" else [args.scene]
    all_results = {}

    for scene in scenes:
        if scene not in SCENES:
            print(f"Unknown scene: {scene}. Available: {list(SCENES.keys())}")
            continue
        result = process_scene(scene, do_pgo=args.do_pgo, max_threads=args.max_threads)
        if result:
            all_results[scene] = result

    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Scene':<12} {'Agent':<7} {'ATE(cm)':<10} {'ATE_al(cm)':<12} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}")
        print("-" * 65)
        for scene, r in all_results.items():
            for aid in [0, 1]:
                ate = r.get(f"agent_{aid}_ate_rmse_cm", float('nan'))
                ate_al = r.get(f"agent_{aid}_ate_aligned_cm", float('nan'))
                psnr = r.get("fine_psnr", float('nan'))
                ssim = r.get("fine_ssim", float('nan'))
                lpips = r.get("fine_lpips", float('nan'))
                print(f"{scene:<12} {aid:<7} {ate:<10.2f} {ate_al:<12.2f} {psnr:<8.2f} {ssim:<8.3f} {lpips:<8.3f}")

    # Save combined results
    summary_path = os.path.join(WORKSPACE, "outputs/baselines/magic_slam/magic_2agent_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {summary_path}")


if __name__ == "__main__":
    main()
