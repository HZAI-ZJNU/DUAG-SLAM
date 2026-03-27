#!/usr/bin/env python3
"""
Test PGO improvements offline on saved trajectories.
No need to re-run SLAM — uses saved est/gt trajectory files.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from core.consensus.lie_algebra import se3_exp, se3_log


def load_tum(path):
    """Load TUM trajectory -> list of 4x4 numpy arrays (c2w)."""
    poses = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses


def compute_ate(est, gt):
    """ATE RMSE between two lists of 4x4 c2w poses."""
    assert len(est) == len(gt)
    errors = []
    for e, g in zip(est, gt):
        diff = np.linalg.inv(g) @ e
        errors.append(np.linalg.norm(diff[:3, 3]))
    return np.sqrt(np.mean(np.array(errors) ** 2))


def detect_loop_closures(gt_poses, distance_thresh=2.0, step=50):
    """Detect inter-agent loop closures from GT."""
    n_agents = len(gt_poses)
    lcs = []
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


def pgo_adam(trajectories, loop_closures, n_iterations=2000,
             odom_weight=1.0, lc_weight=500.0, lr=0.01, keyframe_stride=10):
    """
    Improved PGO with Adam optimizer, proper weights, no clamping.
    """
    agent_ids = sorted(trajectories.keys())
    n_frames = len(trajectories[agent_ids[0]])

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
    print(f"  PGO: {n_kf} keyframes, {len(loop_closures)} LCs")

    init_poses = {}
    for aid in agent_ids:
        for kf in keyframe_indices:
            init_poses[(aid, kf)] = torch.from_numpy(
                trajectories[aid][kf]).float().detach()

    corrections = {}
    for key in init_poses:
        aid, t = key
        if t == 0:
            corrections[key] = torch.zeros(6)
        else:
            corrections[key] = torch.zeros(6, requires_grad=True)

    opt_params = [v for v in corrections.values() if v.requires_grad]
    if not opt_params:
        return trajectories

    optimizer = torch.optim.Adam(opt_params, lr=lr)

    odom_edges = []
    for aid in agent_ids:
        for i in range(n_kf - 1):
            kf_a = keyframe_indices[i]
            kf_b = keyframe_indices[i + 1]
            T_a = init_poses[(aid, kf_a)]
            T_b = init_poses[(aid, kf_b)]
            z_odom = torch.linalg.inv(T_a) @ T_b
            # Weight inversely proportional to frame gap (more uncertain over longer gaps)
            gap = kf_b - kf_a
            weight = odom_weight / max(1, gap / keyframe_stride)
            odom_edges.append((aid, kf_a, aid, kf_b, z_odom.detach(), weight))

    lc_edges = []
    for lc_frame, ai, aj, T_rel_np in loop_closures:
        if lc_frame >= n_frames:
            continue
        T_rel = torch.from_numpy(T_rel_np).float().detach()
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
        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=5.0)
        optimizer.step()

        cost_val = total_cost.item()
        if cost_val < best_cost:
            best_cost = cost_val
            best_corrections = {k: v.clone().detach() for k, v in corrections.items()}

        if iteration % 200 == 0 or iteration == n_iterations - 1:
            print(f"    iter {iteration}: cost = {cost_val:.4f}")

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
                init_T = torch.from_numpy(trajectories[aid][t]).float()
                pose = (init_T @ se3_exp(corr)).numpy()
            corrected[aid].append(pose)

    return corrected


def pgo_direct_align(trajectories, gt_trajectories, loop_closures):
    """
    Direct SE(3) alignment using loop closure frames.
    At each LC frame, we know the GT relative pose and can compute
    the required correction. Distribute it smoothly.
    """
    agent_ids = sorted(trajectories.keys())
    n_frames = len(trajectories[agent_ids[0]])

    # First, try Sim(3) alignment per agent using all GT correspondences
    corrected = {}
    for aid in agent_ids:
        est = trajectories[aid]
        gt = gt_trajectories[aid]

        # Collect position correspondences
        est_pts = np.array([e[:3, 3] for e in est])
        gt_pts = np.array([g[:3, 3] for g in gt])

        # Solve for optimal rigid alignment (Umeyama)
        R_align, t_align, s_align = umeyama_alignment(est_pts, gt_pts)

        corrected[aid] = []
        for t in range(n_frames):
            T = est[t].copy()
            T[:3, 3] = s_align * R_align @ T[:3, 3] + t_align
            T[:3, :3] = R_align @ T[:3, :3]
            corrected[aid].append(T)

    return corrected


def umeyama_alignment(src, dst):
    """Umeyama alignment: find R, t, s such that dst ≈ s * R @ src + t."""
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = np.mean(np.sum(src_c ** 2, axis=1))
    H = (src_c.T @ dst_c) / n

    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = Vt.T @ S @ U.T
    s = np.trace(np.diag(D) @ S) / sigma_src
    t = mu_dst - s * R @ mu_src
    return R, t, s


def main():
    base = "/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM"
    traj_dir = f"{base}/outputs/trajectories/duag_c_apart0_2500_no_consensus"

    # Load trajectories
    est = {
        0: load_tum(f"{traj_dir}/robot_0_est.txt"),
        1: load_tum(f"{traj_dir}/robot_1_est.txt"),
    }
    gt = {
        0: load_tum(f"{traj_dir}/robot_0_gt.txt"),
        1: load_tum(f"{traj_dir}/robot_1_gt.txt"),
    }

    print("=== Raw (no-consensus) ===")
    for aid in [0, 1]:
        ate = compute_ate(est[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate:.6f} m = {ate*100:.4f} cm")

    # Detect loop closures
    gt_np = {k: v for k, v in gt.items()}
    lcs = detect_loop_closures(gt_np, distance_thresh=2.0)
    print(f"\n  {len(lcs)} loop closures detected")
    for f, ai, aj, _ in lcs[:10]:
        print(f"    Frame {f}: agent {ai} <-> agent {aj}")

    # Test 1: Sim(3) alignment (upper bound — uses all GT frames)
    print("\n=== Sim(3) alignment (using ALL GT frames — upper bound) ===")
    aligned = pgo_direct_align(est, gt, lcs)
    for aid in [0, 1]:
        ate = compute_ate(aligned[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate:.6f} m = {ate*100:.4f} cm")

    # Test 2: Original PGO (reproduce the issue)
    print("\n=== Original PGO (odom=100, lc=50, SGD) ===")
    from core.consensus.trajectory_pgo import simple_pgo
    lc_tuples = [(f, ai, aj, T) for f, ai, aj, T in lcs]
    result_old = simple_pgo(est, lc_tuples, n_iterations=300,
                            odom_weight=100.0, lc_weight=50.0)
    for aid in [0, 1]:
        ate = compute_ate(result_old[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate:.6f} m = {ate*100:.4f} cm")

    # Test 3: Fixed PGO (Adam, proper weights)
    print("\n=== Fixed PGO (odom=1, lc=500, Adam, 2000 iters) ===")
    result_new = pgo_adam(est, lc_tuples, n_iterations=2000,
                          odom_weight=1.0, lc_weight=500.0, lr=0.01)
    for aid in [0, 1]:
        ate = compute_ate(result_new[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate:.6f} m = {ate*100:.4f} cm")

    # Test 4: Even stronger LC weight
    print("\n=== Strong PGO (odom=0.1, lc=2000, Adam, 3000 iters) ===")
    result_strong = pgo_adam(est, lc_tuples, n_iterations=3000,
                             odom_weight=0.1, lc_weight=2000.0, lr=0.005)
    for aid in [0, 1]:
        ate = compute_ate(result_strong[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate:.6f} m = {ate*100:.4f} cm")

    # Test 5: Denser keyframes
    print("\n=== Dense PGO (stride=1, odom=1, lc=1000, Adam) ===")
    result_dense = pgo_adam(est, lc_tuples, n_iterations=2000,
                            odom_weight=1.0, lc_weight=1000.0, lr=0.005,
                            keyframe_stride=1)
    for aid in [0, 1]:
        ate = compute_ate(result_dense[aid], gt[aid])
        print(f"  Agent {aid}: ATE = {ate:.6f} m = {ate*100:.4f} cm")


if __name__ == "__main__":
    main()
