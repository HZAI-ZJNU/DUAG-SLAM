#!/usr/bin/env python3
"""
Recovery script: load saved submaps from crashed 2-agent MAGiC-SLAM run,
then run loop detection + PGO + map merging + evaluation in the main process.

This avoids the multiprocessing-related buffer overflow.

Usage:
    cd repos/MAGiC-SLAM
    python ../../scripts/magic_pgo_recovery.py <config_yaml> [--skip-refine]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add MAGiC-SLAM to path
sys.path.insert(0, ".")

from src.entities.datasets import get_dataset
from src.entities.loop_detection.loop_detector import LoopDetector
from src.entities.pose_graph_adapter import PoseGraphAdapter
from src.utils.io_utils import load_config, save_dict_to_ckpt, save_dict_to_json
from src.utils.magic_slam_utils import (
    apply_pose_correction,
    merge_submaps,
    refine_map,
    register_agents_submaps,
    register_submaps,
)
from src.utils.mapping_eval import eval_agents_rendering
from src.utils import vis_utils


def load_agent_submaps(output_path: Path, agent_id: int):
    """Load all saved submaps for an agent."""
    submap_dir = output_path / f"agent_{agent_id}" / "submaps"
    submaps = []
    for ckpt in sorted(submap_dir.glob("*.ckpt")):
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        submaps.append(data)
    return submaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--skip-refine", action="store_true",
                        help="Skip 3000-iter refinement (faster)")
    parser.add_argument("--max-threads", type=int, default=5,
                        help="Threads for ICP registration (reduce if crashing)")
    args = parser.parse_args()

    config = load_config(args.config_path)
    output_path = Path(config["data"]["output_path"])
    agent_ids = config["data"]["agent_ids"]

    print(f"Output path: {output_path}")
    print(f"Agent IDs: {agent_ids}")

    # Load submaps from saved checkpoints
    agents_submaps = {}
    for aid in agent_ids:
        submaps = load_agent_submaps(output_path, aid)
        agents_submaps[aid] = submaps
        print(f"Agent {aid}: loaded {len(submaps)} submaps")

    # === Loop Detection ===
    print("\n=== Loop Detection ===")
    loop_detector = LoopDetector(config["loop_detection"])
    print("LoopDetector created successfully")

    agents_c2ws = {}
    for agent_id, agent_submaps in agents_submaps.items():
        agents_c2ws[agent_id] = np.vstack(
            [submap["submap_c2ws"] for submap in agent_submaps]
        )

    intra_loops, inter_loops = loop_detector.detect_loops(agents_submaps)
    print(f"Intra loops: {len(intra_loops)}, Inter loops: {len(inter_loops)}")

    vis_utils.plot_agents_pose_graph(
        agents_c2ws, {}, intra_loops, inter_loops,
        output_path=str(output_path / "all_loops.png"),
    )

    # === Registration (ICP) ===
    print("\n=== ICP Registration ===")
    intra_loops = register_agents_submaps(
        agents_submaps, intra_loops, register_submaps,
        max_threads=args.max_threads,
    )
    inter_loops = register_agents_submaps(
        agents_submaps, inter_loops, register_submaps,
        max_threads=args.max_threads,
    )

    intra_loops = loop_detector.filter_loops(intra_loops)
    inter_loops = loop_detector.filter_loops(inter_loops)
    print(f"After filtering — intra: {len(intra_loops)}, inter: {len(inter_loops)}")

    # === Pose Graph Optimization ===
    print("\n=== Pose Graph Optimization ===")
    graph_wrapper = PoseGraphAdapter(agents_submaps, intra_loops + inter_loops)
    if len(intra_loops + inter_loops) > 0:
        graph_wrapper.optimize()
        print("PGO done")
    else:
        print("No loops found — skipping PGO")

    agents_opt_kf_c2ws = apply_pose_correction(
        graph_wrapper.get_poses(), agents_submaps
    )

    # Save optimized poses
    for aid in agent_ids:
        save_dict_to_ckpt(
            torch.from_numpy(agents_opt_kf_c2ws[aid]).float(),
            "optimized_kf_c2w.ckpt",
            directory=output_path / f"agent_{aid}",
        )

    # Also load+save kf_ids
    agents_kf_ids = {}
    for agent_id, agent_submaps in agents_submaps.items():
        kf_ids = np.empty(0)
        for submap in agent_submaps:
            kf_ids = np.concatenate([kf_ids, submap["keyframe_ids"]])
        agents_kf_ids[agent_id] = kf_ids.astype(int)
        save_dict_to_ckpt(
            torch.from_numpy(agents_kf_ids[agent_id]),
            "kf_ids.ckpt",
            directory=output_path / f"agent_{aid}",
        )

    # Load estimated c2ws for ATE computation
    for aid in agent_ids:
        est_path = output_path / f"agent_{aid}" / "estimated_c2w.ckpt"
        if est_path.exists():
            est_c2ws = torch.load(est_path, map_location="cpu", weights_only=False)
            print(f"Agent {aid}: estimated_c2w shape = {est_c2ws.shape}")

    # === Map Merging ===
    print("\n=== Map Merging ===")
    # We need opt_args from a mapper; create a simple default
    from src.entities.mapper import Mapper
    mapper_config = config.get("mapping", {})
    # Create a temporary agent to get opt_args
    from argparse import Namespace
    opt_args = Namespace(
        percent_dense=0.01,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
    )

    start = time.time()
    merged_map = merge_submaps(agents_submaps, agents_kf_ids, agents_opt_kf_c2ws, opt_args)
    merge_time = time.time() - start
    print(f"Merge time: {merge_time:.1f}s")

    merged_map.save_ply(str(output_path / "merged_coarse.ply"))
    print(f"Saved coarse PLY: {output_path / 'merged_coarse.ply'}")

    # === Evaluation ===
    print("\n=== Coarse Map Evaluation ===")
    # Load datasets for evaluation
    input_path = Path(config["data"]["input_path"])
    scene_name = config["data"]["scene_name"]
    dataset_name = config["dataset_name"]

    agents_datasets = {}
    for aid in agent_ids:
        agent_input = sorted(input_path.glob("*"))[aid]
        dataset_config = {
            **{"input_path": str(agent_input), "dataset_name": dataset_name},
            **config["cam"],
        }
        if "frame_limit" in config["data"]:
            dataset_config["frame_limit"] = config["data"]["frame_limit"]
        agents_datasets[aid] = get_dataset(dataset_name)(dataset_config)

    coarse_psnr, coarse_lpips, coarse_ssim, coarse_dl1 = eval_agents_rendering(
        merged_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws
    )
    save_dict_to_json(
        {"psnr": coarse_psnr, "lpips": coarse_lpips,
         "ssim": coarse_ssim, "depth_l1": coarse_dl1},
        "coarse_render_metrics.json",
        directory=output_path,
    )
    print(f"Coarse — PSNR: {np.mean(coarse_psnr):.3f}, SSIM: {np.mean(coarse_ssim):.4f}, "
          f"LPIPS: {np.mean(coarse_lpips):.4f}, Depth_L1: {np.mean(coarse_dl1):.4f}")

    if not args.skip_refine:
        print("\n=== Refinement (3000 iters) ===")
        start = time.time()
        refined_map = refine_map(
            merged_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws,
            iterations=3000,
        )
        refine_time = time.time() - start
        print(f"Refine time: {refine_time:.1f}s")

        refined_map.save_ply(str(output_path / "merged_refined.ply"))

        fine_psnr, fine_lpips, fine_ssim, fine_dl1 = eval_agents_rendering(
            refined_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws
        )
        save_dict_to_json(
            {"psnr": fine_psnr, "lpips": fine_lpips,
             "ssim": fine_ssim, "depth_l1": fine_dl1},
            "fine_render_metrics.json",
            directory=output_path,
        )
        print(f"Refined — PSNR: {np.mean(fine_psnr):.3f}, SSIM: {np.mean(fine_ssim):.4f}, "
              f"LPIPS: {np.mean(fine_lpips):.4f}, Depth_L1: {np.mean(fine_dl1):.4f}")
    else:
        print("Skipping refinement (--skip-refine)")

    # === ATE Computation ===
    print("\n=== Trajectory Evaluation ===")
    for aid in agent_ids:
        est_path = output_path / f"agent_{aid}" / "estimated_c2w.ckpt"
        if not est_path.exists():
            continue
        est_c2ws = torch.load(est_path, map_location="cpu", weights_only=False)

        # Load GT
        agent_input = sorted(input_path.glob("*"))[aid]
        dataset_config = {
            **{"input_path": str(agent_input), "dataset_name": dataset_name},
            **config["cam"],
        }
        if "frame_limit" in config["data"]:
            dataset_config["frame_limit"] = config["data"]["frame_limit"]
        ds = get_dataset(dataset_name)(dataset_config)
        gt_c2ws = np.array(ds.poses)

        # Compute ATE on all frames
        n = min(len(est_c2ws), len(gt_c2ws))
        est_trans = est_c2ws[:n, :3, 3].numpy()
        gt_trans = gt_c2ws[:n, :3, 3]
        ate_rmse = np.sqrt(np.mean(np.sum((est_trans - gt_trans) ** 2, axis=1)))
        print(f"Agent {aid}: ATE RMSE = {ate_rmse:.4f} m  ({ate_rmse*100:.2f} cm)")

        # Save optimized kf ATE
        if aid in agents_opt_kf_c2ws and aid in agents_kf_ids:
            kf_ids = agents_kf_ids[aid]
            opt_kf = agents_opt_kf_c2ws[aid]
            gt_kf = gt_c2ws[kf_ids]
            n_kf = min(len(opt_kf), len(gt_kf))
            opt_trans = opt_kf[:n_kf, :3, 3]
            gt_kf_trans = gt_kf[:n_kf, :3, 3]
            ate_kf = np.sqrt(np.mean(np.sum((opt_trans - gt_kf_trans) ** 2, axis=1)))
            print(f"Agent {aid}: Optimized KF ATE RMSE = {ate_kf:.4f} m  ({ate_kf*100:.2f} cm)")

    print("\n=== Recovery Complete ===")


if __name__ == "__main__":
    main()
