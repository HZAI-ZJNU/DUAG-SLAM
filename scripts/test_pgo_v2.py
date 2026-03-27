#!/usr/bin/env python3
"""
Analyze drift pattern and test proper PGO on saved trajectories.
Key approach:
1. SE(3)-align first (like compute_ate_rmse does)
2. Analyze per-frame drift pattern
3. Apply PGO on aligned trajectories
4. Test with different loop closure density
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from core.consensus.lie_algebra import se3_exp, se3_log


def load_tum(path):
    poses = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8: continue
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses


def se3_align(est_pos, gt_pos):
    """SE(3) Umeyama alignment (rotation + translation, no scale). Returns R, t."""
    gt_mean = gt_pos.mean(axis=0)
    est_mean = est_pos.mean(axis=0)
    gt_c = gt_pos - gt_mean
    est_c = est_pos - est_mean
    H = est_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = gt_mean - R @ est_mean
    return R, t


def compute_ate(est, gt):
    n = min(len(est), len(gt))
    gt_pos = np.array([gt[i][:3, 3] for i in range(n)])
    est_pos = np.array([est[i][:3, 3] for i in range(n)])
    R, t = se3_align(est_pos, gt_pos)
    est_aligned = (R @ est_pos.T).T + t
    errors = np.linalg.norm(gt_pos - est_aligned, axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), errors, R, t


def align_trajectory(traj, R, t):
    """Apply SE(3) alignment to a list of c2w poses."""
    aligned = []
    for T in traj:
        T_new = T.copy()
        T_new[:3, 3] = R @ T[:3, 3] + t
        T_new[:3, :3] = R @ T[:3, :3]
        aligned.append(T_new)
    return aligned


def detect_loop_closures_dense(gt_poses, distance_thresh=2.0, step=1):
    """Detect ALL inter-agent loop closures from GT."""
    lcs = []
    n_agents = len(gt_poses)
    for ai in range(n_agents):
        for aj in range(ai + 1, n_agents):
            n_min = min(len(gt_poses[ai]), len(gt_poses[aj]))
            for k in range(0, n_min, step):
                T_i = gt_poses[ai][k]
                T_j = gt_poses[aj][k]
                dist = np.linalg.norm(T_i[:3, 3] - T_j[:3, 3])
                if dist < distance_thresh:
                    T_rel = T_i @ np.linalg.inv(T_j)
                    lcs.append((k, ai, aj, T_rel))
    return lcs


def pgo_aligned(trajectories, gt_trajectories, loop_closures,
                n_iterations=3000, odom_weight=1.0, lc_weight=500.0,
                lr=0.01, keyframe_stride=5):
    """
    PGO that first SE(3)-aligns, then applies corrections.
    """
    agent_ids = sorted(trajectories.keys())
    n_frames = len(trajectories[agent_ids[0]])

    # Step 1: SE(3)-align each trajectory
    aligned_trajs = {}
    for aid in agent_ids:
        gt_pos = np.array([gt_trajectories[aid][i][:3, 3] for i in range(n_frames)])
        est_pos = np.array([trajectories[aid][i][:3, 3] for i in range(n_frames)])
        R, t = se3_align(est_pos, gt_pos)
        aligned_trajs[aid] = align_trajectory(trajectories[aid], R, t)

    # Step 2: Compute per-frame error AFTER alignment
    for aid in agent_ids:
        errors = [np.linalg.norm(aligned_trajs[aid][i][:3, 3] - gt_trajectories[aid][i][:3, 3])
                  for i in range(n_frames)]
        print(f"    Agent {aid} after alignment: mean={np.mean(errors)*100:.3f}cm, max={np.max(errors)*100:.3f}cm")

    # Step 3: PGO on aligned trajectories
    lc_frame_set = set()
    for lc_frame, ai, aj, _ in loop_closures:
        if lc_frame < n_frames:
            lc_frame_set.add(lc_frame)

    keyframe_indices = sorted(set(
        list(range(0, n_frames, keyframe_stride)) +
        [n_frames - 1] +
        list(lc_frame_set)
    ))
    n_kf = len(keyframe_indices)
    print(f"    PGO: {n_kf} keyframes, {len(loop_closures)} LCs, "
          f"kf_stride={keyframe_stride}")

    init_poses = {}
    for aid in agent_ids:
        for kf in keyframe_indices:
            init_poses[(aid, kf)] = torch.from_numpy(
                aligned_trajs[aid][kf]).float().detach()

    corrections = {}
    for key in init_poses:
        aid, t = key
        if t == 0:
            corrections[key] = torch.zeros(6)
        else:
            corrections[key] = torch.zeros(6, requires_grad=True)

    opt_params = [v for v in corrections.values() if v.requires_grad]
    if not opt_params:
        return aligned_trajs

    optimizer = torch.optim.Adam(opt_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations, eta_min=lr*0.01)

    # Odometry edges with gap-dependent weighting
    odom_edges = []
    for aid in agent_ids:
        for i in range(n_kf - 1):
            kf_a = keyframe_indices[i]
            kf_b = keyframe_indices[i + 1]
            T_a = init_poses[(aid, kf_a)]
            T_b = init_poses[(aid, kf_b)]
            z_odom = torch.linalg.inv(T_a) @ T_b
            gap = kf_b - kf_a
            # Weigh inversely with √gap — longer gaps = more uncertainty
            w = odom_weight / np.sqrt(max(1, gap))
            odom_edges.append((aid, kf_a, aid, kf_b, z_odom.detach(), w))

    # Loop closure edges (from GT)
    lc_edges = []
    for lc_frame, ai, aj, T_rel_np in loop_closures:
        if lc_frame >= n_frames:
            continue
        # T_rel for ALIGNED trajectories (not original)
        T_i_aligned = aligned_trajs[ai][lc_frame]
        T_j_aligned = aligned_trajs[aj][lc_frame]
        T_rel_aligned = T_i_aligned @ np.linalg.inv(T_j_aligned)
        T_rel = torch.from_numpy(T_rel_aligned).float().detach()
        lc_edges.append((ai, lc_frame, aj, lc_frame, T_rel))

    best_cost = float('inf')
    best_corrections = {k: v.clone().detach() for k, v in corrections.items()}

    for iteration in range(n_iterations):
        optimizer.zero_grad()
        total_cost = torch.tensor(0.0)

        def get_pose(key):
            return init_poses[key] @ se3_exp(corrections[key])

        for aid_a, t_a, aid_b, t_b, z_odom, w in odom_edges:
            T_a = get_pose((aid_a, t_a))
            T_b = get_pose((aid_b, t_b))
            T_pred = torch.linalg.inv(T_a) @ T_b
            residual = se3_log(torch.linalg.inv(z_odom) @ T_pred)
            total_cost = total_cost + w * residual.pow(2).sum()

        for ai, ti, aj, tj, T_rel in lc_edges:
            T_a = get_pose((ai, ti))
            T_b = get_pose((aj, tj))
            T_est_rel = T_a @ torch.linalg.inv(T_b)
            residual = se3_log(T_est_rel @ torch.linalg.inv(T_rel))
            total_cost = total_cost + lc_weight * residual.pow(2).sum()

        total_cost.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=10.0)
        optimizer.step()
        scheduler.step()

        cost_val = total_cost.item()
        if cost_val < best_cost:
            best_cost = cost_val
            best_corrections = {k: v.clone().detach() for k, v in corrections.items()}

        if iteration % 500 == 0 or iteration == n_iterations - 1:
            print(f"      iter {iteration}: cost = {cost_val:.4f}")

    corrections = best_corrections

    corrected = {}
    for aid in agent_ids:
        corrected[aid] = []
        for t in range(n_frames):
            if (aid, t) in corrections:
                pose = (init_poses[(aid, t)] @ se3_exp(corrections[(aid, t)])).numpy()
            else:
                kf_before = max(kf for kf in keyframe_indices if kf <= t)
                kf_after = min(kf for kf in keyframe_indices if kf >= t)
                if kf_before == kf_after:
                    corr = corrections[(aid, kf_before)]
                else:
                    alpha = float(t - kf_before) / float(kf_after - kf_before)
                    corr_a = corrections[(aid, kf_before)]
                    corr_b = corrections[(aid, kf_after)]
                    corr = corr_a * (1 - alpha) + corr_b * alpha
                init_T = torch.from_numpy(aligned_trajs[aid][t]).float()
                pose = (init_T @ se3_exp(corr)).numpy()
            corrected[aid].append(pose)

    return corrected


def main():
    base = "/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM"
    traj_dir = f"{base}/outputs/trajectories/duag_c_apart0_2500_no_consensus"

    est = {
        0: load_tum(f"{traj_dir}/robot_0_est.txt"),
        1: load_tum(f"{traj_dir}/robot_1_est.txt"),
    }
    gt = {
        0: load_tum(f"{traj_dir}/robot_0_gt.txt"),
        1: load_tum(f"{traj_dir}/robot_1_gt.txt"),
    }
    n_frames = len(est[0])

    # === Analysis ===
    print("=" * 60)
    print("DRIFT ANALYSIS (no-consensus, 2500 frames)")
    print("=" * 60)

    for aid in [0, 1]:
        ate, errors, R, t = compute_ate(est[aid], gt[aid])
        print(f"\nAgent {aid}: ATE = {ate*100:.4f} cm")
        # Show drift pattern: error at different points
        for pct in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            idx = min(int(pct * n_frames) - 1, n_frames - 1)
            print(f"  Frame {idx} ({pct*100:.0f}%): error = {errors[idx]*100:.4f} cm")

    # === Check agent proximity across all frames ===
    print(f"\n{'='*60}")
    print("INTER-AGENT PROXIMITY (GT)")
    print(f"{'='*60}")
    dists = []
    close_frames = []
    for k in range(n_frames):
        d = np.linalg.norm(gt[0][k][:3, 3] - gt[1][k][:3, 3])
        dists.append(d)
        if d < 2.0:
            close_frames.append(k)
    print(f"  Min distance: {min(dists):.3f} m at frame {np.argmin(dists)}")
    print(f"  Max distance: {max(dists):.3f} m")
    print(f"  Frames within 2m: {len(close_frames)} / {n_frames}")
    if close_frames:
        print(f"  Close frame range: {close_frames[0]} - {close_frames[-1]}")

    # Try different distance thresholds
    for thresh in [1.0, 2.0, 3.0, 5.0, 10.0]:
        n_close = sum(1 for d in dists if d < thresh)
        print(f"  Frames within {thresh}m: {n_close}")

    # === PGO Tests ===
    print(f"\n{'='*60}")
    print("PGO TESTS")
    print(f"{'='*60}")

    # Test 1: Sparse LCs (step=50, original)
    lcs_sparse = detect_loop_closures_dense(gt, distance_thresh=2.0, step=50)
    print(f"\n--- Test 1: Sparse LC (step=50, {len(lcs_sparse)} LCs) ---")
    result1 = pgo_aligned(est, gt, lcs_sparse,
                          n_iterations=2000, odom_weight=1.0, lc_weight=500.0,
                          lr=0.005, keyframe_stride=10)
    for aid in [0, 1]:
        ate, _, _, _ = compute_ate(result1[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate*100:.4f} cm")

    # Test 2: Dense LCs (step=1, more constraints)
    lcs_dense = detect_loop_closures_dense(gt, distance_thresh=2.0, step=1)
    print(f"\n--- Test 2: Dense LC (step=1, {len(lcs_dense)} LCs) ---")
    result2 = pgo_aligned(est, gt, lcs_dense,
                          n_iterations=2000, odom_weight=0.5, lc_weight=1000.0,
                          lr=0.005, keyframe_stride=5)
    for aid in [0, 1]:
        ate, _, _, _ = compute_ate(result2[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate*100:.4f} cm")

    # Test 3: Very large threshold for more LC coverage
    lcs_wide = detect_loop_closures_dense(gt, distance_thresh=5.0, step=10)
    print(f"\n--- Test 3: Wide threshold (5m, step=10, {len(lcs_wide)} LCs) ---")
    result3 = pgo_aligned(est, gt, lcs_wide,
                          n_iterations=2000, odom_weight=0.5, lc_weight=500.0,
                          lr=0.005, keyframe_stride=5)
    for aid in [0, 1]:
        ate, _, _, _ = compute_ate(result3[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate*100:.4f} cm")

    # Test 4: Ultra-dense with all available frames + large threshold
    lcs_ultra = detect_loop_closures_dense(gt, distance_thresh=10.0, step=1)
    print(f"\n--- Test 4: Ultra-dense (10m, step=1, {len(lcs_ultra)} LCs) ---")
    if len(lcs_ultra) > 100:
        result4 = pgo_aligned(est, gt, lcs_ultra,
                              n_iterations=3000, odom_weight=0.1, lc_weight=500.0,
                              lr=0.003, keyframe_stride=5)
        for aid in [0, 1]:
            ate, _, _, _ = compute_ate(result4[aid], gt[aid])
            print(f"  Agent {aid}: ATE = {ate*100:.4f} cm")
    else:
        print("  Not enough LCs")


if __name__ == "__main__":
    main()
