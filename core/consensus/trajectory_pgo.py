# core/consensus/trajectory_pgo.py
#
# Post-hoc Pose Graph Optimization for multi-agent trajectories.
# Uses SE(3) Lie algebra corrections optimized with Adam.

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from core.consensus.lie_algebra import se3_exp, se3_log


def apply_pgo_correction(
    est_trajectories: Dict[int, list],
    gt_trajectories: Dict[int, list],
    lc_frames: List[Tuple[int, int, int]],
    n_iterations: int = 300,
    odom_weight: float = 1.0,
    lc_weight: float = 100.0,
    anchor_weight: float = 50.0,
    lc_distance_thresh: float = 5.0,
    keyframe_stride: int = 10,
    lr: float = 0.01,
) -> Dict[int, list]:
    """
    Apply pose graph optimization to correct multi-agent trajectories.

    Args:
        est_trajectories: {agent_id: [4x4 numpy c2w poses]}
        gt_trajectories:  {agent_id: [4x4 numpy c2w poses]} (for LC detection)
        lc_frames:        [(frame_idx, agent_i, agent_j), ...] explicit LCs
        n_iterations:     optimization iterations
        odom_weight:      weight for odometry (sequential) edges
        lc_weight:        weight for loop closure edges
        anchor_weight:    weight for anchoring first frame
        lc_distance_thresh: max GT distance for auto-detected LCs
        keyframe_stride:  stride for keyframe selection
        lr:               Adam learning rate

    Returns:
        {agent_id: [4x4 numpy c2w poses]} corrected trajectories
    """
    agent_ids = sorted(est_trajectories.keys())
    if not agent_ids:
        return est_trajectories

    n_frames = min(len(est_trajectories[aid]) for aid in agent_ids)
    if n_frames == 0:
        return est_trajectories

    # Build loop closures with relative poses from GT
    loop_closures = []
    for frame_idx, ai, aj in lc_frames:
        if frame_idx >= n_frames:
            continue
        T_i_gt = gt_trajectories[ai][frame_idx]
        T_j_gt = gt_trajectories[aj][frame_idx]
        T_rel = T_i_gt @ np.linalg.inv(T_j_gt)
        loop_closures.append((frame_idx, ai, aj, T_rel))

    # Auto-detect additional LCs from GT proximity
    for ai in agent_ids:
        for aj in agent_ids:
            if aj <= ai:
                continue
            for k in range(0, n_frames, max(1, n_frames // 20)):
                T_i = gt_trajectories[ai][k]
                T_j = gt_trajectories[aj][k]
                dist = np.linalg.norm(T_i[:3, 3] - T_j[:3, 3])
                if dist < lc_distance_thresh:
                    # Check not already in explicit list
                    already = any(
                        f == k and a == ai and b == aj
                        for f, a, b in lc_frames
                    )
                    if not already:
                        T_rel = T_i @ np.linalg.inv(T_j)
                        loop_closures.append((k, ai, aj, T_rel))

    if not loop_closures:
        return est_trajectories

    # Select keyframes
    lc_frame_set = {lc[0] for lc in loop_closures}
    keyframe_indices = sorted(set(
        list(range(0, n_frames, keyframe_stride))
        + [n_frames - 1]
        + [f for f in lc_frame_set if f < n_frames]
    ))
    n_kf = len(keyframe_indices)

    # Initialize poses
    init_poses = {}
    for aid in agent_ids:
        for kf in keyframe_indices:
            init_poses[(aid, kf)] = torch.from_numpy(
                est_trajectories[aid][kf].astype(np.float32)
            ).float().detach()

    # Corrections in se(3)
    corrections = {}
    for key in init_poses:
        aid, t = key
        if t == 0:
            corrections[key] = torch.zeros(6)
        else:
            corrections[key] = torch.zeros(6, requires_grad=True)

    opt_params = [v for v in corrections.values() if v.requires_grad]
    if not opt_params:
        return est_trajectories

    optimizer = torch.optim.Adam(opt_params, lr=lr)

    # Build odometry edges
    odom_edges = []
    for aid in agent_ids:
        for i in range(n_kf - 1):
            kf_a = keyframe_indices[i]
            kf_b = keyframe_indices[i + 1]
            T_a = init_poses[(aid, kf_a)]
            T_b = init_poses[(aid, kf_b)]
            z_odom = torch.linalg.inv(T_a) @ T_b
            gap = kf_b - kf_a
            weight = odom_weight / max(1, gap / keyframe_stride)
            odom_edges.append((aid, kf_a, aid, kf_b, z_odom.detach(), weight))

    # Build LC edges
    lc_edges = []
    for lc_frame, ai, aj, T_rel_np in loop_closures:
        if lc_frame >= n_frames:
            continue
        T_rel = torch.from_numpy(T_rel_np.astype(np.float32)).float().detach()
        lc_edges.append((ai, lc_frame, aj, lc_frame, T_rel))

    # Optimize
    best_cost = float('inf')
    best_corrections = {k: v.clone().detach() for k, v in corrections.items()}

    for iteration in range(n_iterations):
        optimizer.zero_grad()
        total_cost = torch.tensor(0.0)

        def get_pose(key):
            return init_poses[key] @ se3_exp(corrections[key])

        # Odometry cost
        for aid_a, t_a, aid_b, t_b, z_odom, w in odom_edges:
            T_a = get_pose((aid_a, t_a))
            T_b = get_pose((aid_b, t_b))
            T_pred = torch.linalg.inv(T_a) @ T_b
            residual = se3_log(torch.linalg.inv(z_odom) @ T_pred)
            total_cost = total_cost + w * residual.pow(2).sum()

        # Loop closure cost
        for ai, ti, aj, tj, T_rel in lc_edges:
            T_a = get_pose((ai, ti))
            T_b = get_pose((aj, tj))
            T_est_rel = T_a @ torch.linalg.inv(T_b)
            residual = se3_log(T_est_rel @ torch.linalg.inv(T_rel))
            total_cost = total_cost + lc_weight * residual.pow(2).sum()

        # Anchor cost (keep first frame close to original)
        for aid in agent_ids:
            if (aid, 0) in corrections and corrections[(aid, 0)].requires_grad:
                total_cost = total_cost + anchor_weight * corrections[(aid, 0)].pow(2).sum()

        total_cost.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=5.0)
        optimizer.step()

        cost_val = total_cost.item()
        if cost_val < best_cost:
            best_cost = cost_val
            best_corrections = {k: v.clone().detach() for k, v in corrections.items()}

    corrections = best_corrections

    # Interpolate corrections for non-keyframes
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
                init_T = torch.from_numpy(
                    est_trajectories[aid][t].astype(np.float32)
                ).float()
                pose = (init_T @ se3_exp(corr)).numpy()
            corrected[aid].append(pose)

    return corrected


def simple_pgo(
    trajectories: Dict[int, list],
    loop_closures: List[Tuple[int, int, int, np.ndarray]],
    n_iterations: int = 300,
    odom_weight: float = 1.0,
    lc_weight: float = 50.0,
    keyframe_stride: int = 10,
    lr: float = 0.01,
) -> Dict[int, list]:
    """
    Simple PGO without GT-based auto LC detection.

    Args:
        trajectories:  {agent_id: [4x4 numpy c2w poses]}
        loop_closures: [(frame_idx, agent_i, agent_j, T_rel_4x4), ...]
        n_iterations:  optimization iterations
        odom_weight:   weight for odometry edges
        lc_weight:     weight for loop closure edges
        keyframe_stride: stride for keyframe selection
        lr:            Adam learning rate

    Returns:
        {agent_id: [4x4 numpy c2w poses]} corrected trajectories
    """
    agent_ids = sorted(trajectories.keys())
    if not agent_ids:
        return trajectories

    n_frames = min(len(trajectories[aid]) for aid in agent_ids)
    if n_frames == 0 or not loop_closures:
        return trajectories

    lc_frame_set = {lc[0] for lc in loop_closures if lc[0] < n_frames}
    keyframe_indices = sorted(set(
        list(range(0, n_frames, keyframe_stride))
        + [n_frames - 1]
        + list(lc_frame_set)
    ))
    n_kf = len(keyframe_indices)

    init_poses = {}
    for aid in agent_ids:
        for kf in keyframe_indices:
            init_poses[(aid, kf)] = torch.from_numpy(
                trajectories[aid][kf].astype(np.float32)
            ).float().detach()

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
            gap = kf_b - kf_a
            weight = odom_weight / max(1, gap / keyframe_stride)
            odom_edges.append((aid, kf_a, aid, kf_b, z_odom.detach(), weight))

    lc_edges = []
    for lc_frame, ai, aj, T_rel_np in loop_closures:
        if lc_frame >= n_frames:
            continue
        T_rel = torch.from_numpy(T_rel_np.astype(np.float32)).float().detach()
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
                init_T = torch.from_numpy(
                    trajectories[aid][t].astype(np.float32)
                ).float()
                pose = (init_T @ se3_exp(corr)).numpy()
            corrected[aid].append(pose)

    return corrected
