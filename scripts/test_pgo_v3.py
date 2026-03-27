#!/usr/bin/env python3
"""
Fast PGO test on saved no-consensus trajectories.
Tests whether PGO can reduce ATE below 0.10cm given LC constraints.

Approach: position-only PGO solved as sparse linear system (closed-form).
No iteration needed — exact Cholesky solve.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import torch


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def load_poses(path):
    """Load a trajectory file (TUM .txt or .npy) → list of [4,4] numpy arrays."""
    if path.endswith('.txt'):
        return load_tum_poses(path)
    data = np.load(path, allow_pickle=True)
    if data.ndim == 3:
        return [data[i] for i in range(data.shape[0])]
    return list(data)


def load_tum_poses(path):
    """Load TUM format: timestamp tx ty tz qx qy qz qw → list of [4,4]."""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses


def se3_alignment(est_positions, gt_positions):
    """
    Compute SE(3) alignment (R, t) such that R @ est + t ≈ gt.
    Umeyama method (no scale).
    Returns R [3,3], t [3].
    """
    est = np.array(est_positions)  # [N, 3]
    gt = np.array(gt_positions)    # [N, 3]

    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)

    est_c = est - mu_est
    gt_c = gt - mu_gt

    H = est_c.T @ gt_c  # [3, 3]
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = mu_gt - R @ mu_est

    return R, t


def apply_alignment(poses, R, t):
    """Apply SE(3) alignment to a list of [4,4] poses."""
    aligned = []
    for T in poses:
        T_new = np.eye(4)
        T_new[:3, :3] = R @ T[:3, :3]
        T_new[:3, 3] = R @ T[:3, 3] + t
        aligned.append(T_new)
    return aligned


def compute_ate(est_poses, gt_poses):
    """Compute ATE RMSE after SE(3) alignment."""
    est_pos = np.array([T[:3, 3] for T in est_poses])
    gt_pos = np.array([T[:3, 3] for T in gt_poses])
    R, t = se3_alignment(est_pos, gt_pos)
    aligned_pos = (R @ est_pos.T).T + t
    errors = np.linalg.norm(aligned_pos - gt_pos, axis=1)
    return np.sqrt(np.mean(errors ** 2)), errors, R, t


def detect_lc_frames(gt_0, gt_1, distance_thresh=5.0, step=1):
    """Detect loop closure frames from GT proximity."""
    n = min(len(gt_0), len(gt_1))
    lc_frames = []
    for k in range(0, n, step):
        pos_0 = gt_0[k][:3, 3]
        pos_1 = gt_1[k][:3, 3]
        dist = np.linalg.norm(pos_0 - pos_1)
        if dist < distance_thresh:
            lc_frames.append(k)
    return lc_frames


# ─────────────────────────────────────────────────────────────
# Position-only PGO (closed-form sparse linear solve)
# ─────────────────────────────────────────────────────────────

def position_pgo_single_agent(
    aligned_positions,  # [N, 3] — SE(3)-aligned estimated positions
    gt_positions,       # [N, 3] — ground truth positions
    anchor_frames,      # list of frame indices where we have constraints
    odom_weight=1.0,
    anchor_weight=100.0,
):
    """
    Solve for position corrections c[t] ∈ R³ that minimize:
      Σ_t w_o ||c[t+1] - c[t]||²                    (smoothness)
    + Σ_{k∈anchors} w_a ||pos_aligned[k] + c[k] - pos_gt[k]||²  (anchor)
    
    This is a sparse linear system Hc = b, solved exactly.
    Variables: c[0..N-1] ∈ R³ (3N total).
    
    Returns: corrected_positions = aligned_positions + c
    """
    N = len(aligned_positions)
    if N == 0:
        return aligned_positions.copy()

    # Build sparse system for each dimension independently (they decouple)
    corrected = aligned_positions.copy()
    
    for dim in range(3):
        # Variables: c[0], c[1], ..., c[N-1]
        H = lil_matrix((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)

        # Odometry smoothness: w_o * (c[t+1] - c[t])² → H and b entries
        for t in range(N - 1):
            H[t, t] += odom_weight
            H[t, t+1] -= odom_weight
            H[t+1, t] -= odom_weight
            H[t+1, t+1] += odom_weight
            # b contributions: zero (c[t+1] - c[t] ≈ 0)

        # Anchor constraints: w_a * (pos_aligned[k] + c[k] - pos_gt[k])²
        # Derivative: w_a * c[k] = w_a * (pos_gt[k] - pos_aligned[k])
        for k in anchor_frames:
            if k < N:
                H[k, k] += anchor_weight
                b[k] += anchor_weight * (gt_positions[k, dim] - aligned_positions[k, dim])

        # Fix first frame: no correction (already aligned)
        H[0, 0] += 1000.0  # strong prior: c[0] ≈ 0

        # Solve
        H_csc = csc_matrix(H)
        c = spsolve(H_csc, b)
        corrected[:, dim] += c

    return corrected


def position_pgo_multi_agent(
    aligned_est,    # {agent_id: [N, 3] aligned positions}
    gt_positions,   # {agent_id: [N, 3] GT positions}
    lc_frames,      # list of frame indices (both agents at same frame)
    odom_weight=1.0,
    lc_weight=100.0,
    anchor_weight=0.0,  # direct GT anchoring (if needed)
):
    """
    Multi-agent PGO with inter-agent loop closure constraints.
    
    For 2 agents with N frames each:
    Variables: c_0[0..N-1], c_1[0..N-1] ∈ R³  (6N total)
    
    Cost:
      Σ_a Σ_t w_o ||(c_a[t+1] - c_a[t])||²               (per-agent smoothness)
    + Σ_k  w_lc ||(pos_0[k]+c_0[k]) - (pos_1[k]+c_1[k]) - delta_gt[k]||² (LC)
    
    where delta_gt[k] = gt_0[k] - gt_1[k] (GT relative position at frame k)
    
    Solved as sparse linear system.
    """
    agents = sorted(aligned_est.keys())
    N = min(len(aligned_est[a]) for a in agents)
    n_agents = len(agents)
    total_vars = n_agents * N  # per dimension

    corrected = {}
    for a in agents:
        corrected[a] = aligned_est[a][:N].copy()

    for dim in range(3):
        H = lil_matrix((total_vars, total_vars), dtype=np.float64)
        b = np.zeros(total_vars, dtype=np.float64)

        # Per-agent odometry smoothness
        for idx_a, a in enumerate(agents):
            offset = idx_a * N
            for t in range(N - 1):
                i, j = offset + t, offset + t + 1
                H[i, i] += odom_weight
                H[i, j] -= odom_weight
                H[j, i] -= odom_weight
                H[j, j] += odom_weight

        # Inter-agent loop closure constraints
        # ||pos_0[k] + c_0[k] - pos_1[k] - c_1[k] - delta_gt[k]||²
        # Let r = c_0[k] - c_1[k] - (delta_gt[k] - (pos_0[k] - pos_1[k]))
        # dH/dc_0[k] = w_lc * r, dH/dc_1[k] = -w_lc * r
        for k in lc_frames:
            if k >= N:
                continue
            # Variables: c_0[k] at index 0*N+k, c_1[k] at index 1*N+k
            i0 = 0 * N + k  # Agent 0's correction at frame k
            i1 = 1 * N + k  # Agent 1's correction at frame k

            delta_gt = gt_positions[agents[0]][k, dim] - gt_positions[agents[1]][k, dim]
            delta_est = aligned_est[agents[0]][k, dim] - aligned_est[agents[1]][k, dim]
            residual_const = delta_gt - delta_est

            # Quadratic: w_lc * (c_0[k] - c_1[k] - residual_const)²
            H[i0, i0] += lc_weight
            H[i0, i1] -= lc_weight
            H[i1, i0] -= lc_weight
            H[i1, i1] += lc_weight
            b[i0] += lc_weight * residual_const
            b[i1] -= lc_weight * residual_const

        # Optional: direct GT anchoring for each agent
        if anchor_weight > 0:
            for idx_a, a in enumerate(agents):
                offset = idx_a * N
                for k in lc_frames:
                    if k >= N:
                        continue
                    idx = offset + k
                    target = gt_positions[a][k, dim] - aligned_est[a][k, dim]
                    H[idx, idx] += anchor_weight
                    b[idx] += anchor_weight * target

        # Fix agent 0, frame 0 (no correction — anchor point)
        H[0, 0] += 10000.0

        # Solve
        H_csc = csc_matrix(H)
        c = spsolve(H_csc, b)

        for idx_a, a in enumerate(agents):
            offset = idx_a * N
            corrected[a][:, dim] += c[offset:offset + N]

    return corrected


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    base = "/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM"

    # Find saved TUM trajectories (no-consensus run)
    traj_dirs = [
        os.path.join(base, "outputs/trajectories/duag_c_apart0_2500_no_consensus"),
        os.path.join(base, "outputs/trajectories/duag_c_apart0_2500"),
    ]
    result_dir = None
    for d in traj_dirs:
        if os.path.exists(os.path.join(d, "robot_0_est.txt")):
            result_dir = d
            break

    if result_dir is None:
        print("No saved trajectories found!")
        return

    print(f"=== PGO v3: Fast position-only PGO ===")
    print(f"Trajectory dir: {result_dir}")

    # Load trajectories
    est = {}
    gt = {}
    for agent_id in [0, 1]:
        est_path = os.path.join(result_dir, f"robot_{agent_id}_est.txt")
        gt_path = os.path.join(result_dir, f"robot_{agent_id}_gt.txt")
        if not os.path.exists(est_path):
            print(f"Missing: {est_path}")
            return
        est[agent_id] = load_poses(est_path)
        gt[agent_id] = load_poses(gt_path)
        print(f"  Agent {agent_id}: {len(est[agent_id])} est, {len(gt[agent_id])} GT frames")

    N = min(len(est[0]), len(est[1]), len(gt[0]), len(gt[1]))
    print(f"  Using {N} frames")

    # Baseline ATE (with alignment)
    print(f"\n=== Baseline ATE (SE(3)-aligned) ===")
    baseline_ate = {}
    aligned_pos = {}
    gt_pos = {}
    for agent_id in [0, 1]:
        ate, errors, R, t = compute_ate(est[agent_id][:N], gt[agent_id][:N])
        baseline_ate[agent_id] = ate
        est_p = np.array([T[:3, 3] for T in est[agent_id][:N]])
        aligned_pos[agent_id] = (R @ est_p.T).T + t
        gt_pos[agent_id] = np.array([T[:3, 3] for T in gt[agent_id][:N]])
        print(f"  Agent {agent_id}: ATE = {ate*100:.4f} cm  "
              f"(max={errors.max()*100:.2f}cm, mean={errors.mean()*100:.2f}cm)")

    # Detect loop closures at various thresholds
    print(f"\n=== Loop closure detection (from GT proximity) ===")
    for thresh in [2.0, 5.0, 10.0, 50.0]:
        lc = detect_lc_frames(gt[0][:N], gt[1][:N], thresh, step=1)
        print(f"  thresh={thresh}m: {len(lc)} LC frames "
              f"({100*len(lc)/N:.0f}% of trajectory)")

    # ─── Test 1: Per-agent PGO with GT anchors ─────────────────
    print(f"\n=== Test 1: Per-agent PGO with GT anchors (upper bound) ===")
    for lc_step, lc_thresh in [(50, 5.0), (10, 5.0), (1, 5.0), (1, 50.0)]:
        lc_frames = detect_lc_frames(gt[0][:N], gt[1][:N], lc_thresh, step=lc_step)
        if not lc_frames:
            continue
        for agent_id in [0, 1]:
            corrected_pos = position_pgo_single_agent(
                aligned_pos[agent_id], gt_pos[agent_id],
                lc_frames, odom_weight=1.0, anchor_weight=100.0)
            corr_errors = np.linalg.norm(corrected_pos - gt_pos[agent_id], axis=1)
            corr_ate = np.sqrt(np.mean(corr_errors**2))
            improvement = (1 - corr_ate / baseline_ate[agent_id]) * 100
            if agent_id == 0:
                print(f"  step={lc_step}, thresh={lc_thresh}m ({len(lc_frames)} anchors):")
            print(f"    Agent {agent_id}: {corr_ate*100:.4f}cm ({improvement:+.1f}%)")

    # ─── Test 2: Multi-agent PGO with LC constraints only ─────
    print(f"\n=== Test 2: Multi-agent PGO (LC relative constraints only, no GT anchor) ===")
    for lc_step, lc_thresh, odom_w, lc_w in [
        (50, 5.0, 1.0, 100.0),
        (10, 5.0, 1.0, 100.0),
        (1, 5.0,  1.0, 100.0),
        (1, 5.0,  0.1, 100.0),
        (1, 50.0, 1.0, 100.0),
    ]:
        lc_frames = detect_lc_frames(gt[0][:N], gt[1][:N], lc_thresh, step=lc_step)
        if not lc_frames:
            continue

        corrected = position_pgo_multi_agent(
            aligned_pos, gt_pos, lc_frames,
            odom_weight=odom_w, lc_weight=lc_w, anchor_weight=0.0)

        label = (f"  step={lc_step}, thresh={lc_thresh}m, "
                 f"odom_w={odom_w}, lc_w={lc_w} ({len(lc_frames)} LCs):")
        print(label)
        for agent_id in [0, 1]:
            # Re-align corrected trajectory and compute ATE
            R2, t2 = se3_alignment(corrected[agent_id], gt_pos[agent_id])
            realigned = (R2 @ corrected[agent_id].T).T + t2
            corr_errors = np.linalg.norm(realigned - gt_pos[agent_id], axis=1)
            corr_ate = np.sqrt(np.mean(corr_errors**2))
            improvement = (1 - corr_ate / baseline_ate[agent_id]) * 100
            print(f"    Agent {agent_id}: {corr_ate*100:.4f}cm ({improvement:+.1f}%)")

    # ─── Test 3: Multi-agent PGO with BOTH LC + GT anchor ─────
    print(f"\n=== Test 3: Multi-agent PGO (LC + GT anchors, best case) ===")
    for lc_step, lc_thresh, odom_w, lc_w, anc_w in [
        (10, 5.0, 1.0, 100.0, 50.0),
        (1,  5.0, 1.0, 100.0, 50.0),
        (1, 50.0, 1.0, 100.0, 50.0),
        (1, 50.0, 0.1, 100.0, 100.0),
    ]:
        lc_frames = detect_lc_frames(gt[0][:N], gt[1][:N], lc_thresh, step=lc_step)
        if not lc_frames:
            continue

        corrected = position_pgo_multi_agent(
            aligned_pos, gt_pos, lc_frames,
            odom_weight=odom_w, lc_weight=lc_w, anchor_weight=anc_w)

        label = (f"  step={lc_step}, thresh={lc_thresh}m, "
                 f"odom_w={odom_w}, lc_w={lc_w}, anc_w={anc_w} ({len(lc_frames)} LCs):")
        print(label)
        for agent_id in [0, 1]:
            R2, t2 = se3_alignment(corrected[agent_id], gt_pos[agent_id])
            realigned = (R2 @ corrected[agent_id].T).T + t2
            corr_errors = np.linalg.norm(realigned - gt_pos[agent_id], axis=1)
            corr_ate = np.sqrt(np.mean(corr_errors**2))
            improvement = (1 - corr_ate / baseline_ate[agent_id]) * 100
            print(f"    Agent {agent_id}: {corr_ate*100:.4f}cm ({improvement:+.1f}%)")

    print(f"\n=== Summary ===")
    print(f"Target: < 0.10 cm")
    print(f"Baseline (no-consensus, aligned): "
          f"Agent 0 = {baseline_ate[0]*100:.4f}cm, "
          f"Agent 1 = {baseline_ate[1]*100:.4f}cm")


if __name__ == "__main__":
    main()
