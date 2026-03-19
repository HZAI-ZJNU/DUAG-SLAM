#!/usr/bin/env python3
"""scripts/collect_results.py — Aggregate all result JSON files into a summary CSV.

Reads:
  - outputs/results/duag_c_*.json           (DUAG-C experiments)
  - outputs/baselines/magic_slam/*/         (MAGiC-SLAM outputs)
  - outputs/baselines/mac_ego3d/*/          (MAC-Ego3D outputs)

Writes:
  - outputs/results/summary.csv
  - outputs/results/baselines_magic_slam.json
  - outputs/results/baselines_mac_ego3d.json
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "outputs" / "results"
BASELINES_DIR = ROOT / "outputs" / "baselines"


def load_duag_results():
    """Load all DUAG-C result JSONs."""
    rows = []
    for jf in sorted(RESULTS_DIR.glob("duag_c_*.json")):
        with open(jf) as f:
            data = json.load(f)
        experiment = jf.stem
        for scene, metrics in data.items():
            ate_0 = metrics.get("agent_0_ATE_RMSE", None)
            ate_1 = metrics.get("agent_1_ATE_RMSE", None)
            ate_avg = None
            if ate_0 is not None and ate_1 is not None:
                ate_avg = (ate_0 + ate_1) / 2
            rows.append({
                "system": experiment,
                "scene": scene,
                "ATE_RMSE_agent0": ate_0,
                "ATE_RMSE_agent1": ate_1,
                "ATE_RMSE_avg": ate_avg,
                "RPE_trans_agent0": metrics.get("agent_0_RPE_trans"),
                "RPE_rot_agent0": metrics.get("agent_0_RPE_rot_deg"),
                "admm_iterations_agent0": metrics.get("agent_0_admm_iterations"),
                "comm_bytes_total": metrics.get("comm_bytes_total"),
                "elapsed_s": metrics.get("elapsed_s"),
                "use_fim_weighting": metrics.get("use_fim_weighting"),
            })
    return rows


def load_magic_slam_results():
    """Parse MAGiC-SLAM outputs (JSON metric files in each scene dir)."""
    rows = []
    baseline_data = {}
    magic_dir = BASELINES_DIR / "magic_slam"
    if not magic_dir.exists():
        return rows, baseline_data

    for scene_dir in sorted(magic_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        entry = {"scene": scene}

        # Read render metrics
        for tag in ("fine_render_metrics", "coarse_render_metrics"):
            mf = scene_dir / f"{tag}.json"
            if mf.exists():
                with open(mf) as f:
                    m = json.load(f)
                prefix = "fine_" if "fine" in tag else "coarse_"
                for k, v in m.items():
                    entry[prefix + k] = v

        # Read walltime
        wf = scene_dir / "walltime_stats.json"
        if wf.exists():
            with open(wf) as f:
                entry.update(json.load(f))

        baseline_data[scene] = entry
        rows.append({
            "system": "magic_slam",
            "scene": scene,
            "ATE_RMSE_agent0": None,  # MAGiC-SLAM doesn't output TUM-style ATE directly
            "ATE_RMSE_agent1": None,
            "ATE_RMSE_avg": None,
            "PSNR": entry.get("fine_psnr"),
            "SSIM": entry.get("fine_ssim"),
            "LPIPS": entry.get("fine_lpips"),
            "elapsed_s": entry.get("total_merge_time"),
        })

    # Save structured JSON
    if baseline_data:
        out = RESULTS_DIR / "baselines_magic_slam.json"
        with open(out, "w") as f:
            json.dump(baseline_data, f, indent=2)
        print(f"Saved {out}")

    return rows, baseline_data


def load_mac_ego3d_results():
    """Parse MAC-Ego3D outputs (look for result files in each scene dir)."""
    rows = []
    baseline_data = {}
    macego_dir = BASELINES_DIR / "mac_ego3d"
    if not macego_dir.exists():
        return rows, baseline_data

    for scene_dir in sorted(macego_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        entry = {"scene": scene}

        # Check for eval results
        rj = scene_dir / "results.json"
        if rj.exists():
            with open(rj) as f:
                entry.update(json.load(f))

        # Check for PLY
        entry["has_ply"] = (scene_dir / "scene.ply").exists()

        # Check for trajectory tensors
        for aid in range(2):
            entry[f"has_agent{aid}_traj"] = (scene_dir / f"Agent{aid}_est.pt").exists()

        baseline_data[scene] = entry
        rows.append({
            "system": "mac_ego3d",
            "scene": scene,
            "ATE_RMSE_agent0": None,
            "ATE_RMSE_agent1": None,
            "ATE_RMSE_avg": None,
            "PSNR": entry.get("psnr"),
            "SSIM": entry.get("ssim"),
            "LPIPS": entry.get("lpips"),
        })

    if baseline_data:
        out = RESULTS_DIR / "baselines_mac_ego3d.json"
        with open(out, "w") as f:
            json.dump(baseline_data, f, indent=2)
        print(f"Saved {out}")

    return rows, baseline_data


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # DUAG-C results
    duag_rows = load_duag_results()
    all_rows.extend(duag_rows)
    print(f"DUAG-C: {len(duag_rows)} entries")

    # MAGiC-SLAM
    magic_rows, _ = load_magic_slam_results()
    all_rows.extend(magic_rows)
    print(f"MAGiC-SLAM: {len(magic_rows)} entries")

    # MAC-Ego3D
    macego_rows, _ = load_mac_ego3d_results()
    all_rows.extend(macego_rows)
    print(f"MAC-Ego3D: {len(macego_rows)} entries")

    if not all_rows:
        print("No results found.")
        sys.exit(0)

    # Collect all column names
    fieldnames = []
    for r in all_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    # Write summary CSV
    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved summary CSV: {csv_path}")
    print(f"Total entries: {len(all_rows)}")


if __name__ == "__main__":
    main()
