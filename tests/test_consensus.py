# tests/test_consensus.py
#
# STEP 11 tests for core/consensus/riemannian_admm.py
# Run with: PYTHONPATH="" conda run -n duag python -m pytest tests/test_consensus.py -v
#
# Required tests:
#   test_admm_two_robots_converge   (< 50 iterations)
#   test_admm_preserves_gt          (atol=1e-3)
#   test_ablation_fim_faster        (FIM iters < uniform iters)
#   test_dual_update_shape          (shape check)

import torch
import pytest
from core.types import PoseGraph, ConsensusState
from core.consensus.riemannian_admm import RiemannianADMM
from core.consensus.lie_algebra import se3_exp, se3_log


def _run_admm_pair(
    T_gt: torch.Tensor,
    T_init_0: torch.Tensor,
    T_init_1: torch.Tensor,
    max_iters: int = 50,
    edge_info_weight: float = 1.0,
    rho_init: float = 1.0,
    pose_lr: float = 0.3,
) -> tuple:
    """
    Run distributed ADMM between 2 robots until convergence or max_iters.

    Both robots share the SAME ground-truth pose T_gt (they observe the same
    location from overlapping viewpoints). The ADMM consensus drives both
    robots' pose estimates toward each other. Each robot also has a local
    "anchor" edge tying it to T_gt (representing the visual-odometry prior).

    Returns (converged, n_iters, final_pose_0, final_pose_1).
    """
    admm_0 = RiemannianADMM(
        robot_id=0, rho_init=rho_init, tol_primal=1e-4,
        pose_lr=pose_lr, device="cpu",
    )
    admm_1 = RiemannianADMM(
        robot_id=1, rho_init=rho_init, tol_primal=1e-4,
        pose_lr=pose_lr, device="cpu",
    )

    admm_0.initialize(T_init_0.clone(), neighbor_ids=[1])
    admm_1.initialize(T_init_1.clone(), neighbor_ids=[0])

    # Anchor info matrix — local odometry prior
    anchor_info = torch.eye(6) * edge_info_weight

    for it in range(max_iters):
        T_0 = admm_0.state.primal_pose.clone()
        T_1 = admm_1.state.primal_pose.clone()

        # Each robot's pose graph has an anchor edge to GT
        # (simulating visual-odometry constraint)
        pg_0 = PoseGraph(
            poses={0: T_0, 99: T_gt},
            robot_id=0,
            edges=[(0, 99, torch.eye(4), anchor_info)],
        )
        pg_1 = PoseGraph(
            poses={0: T_1, 99: T_gt},
            robot_id=1,
            edges=[(0, 99, torch.eye(4), anchor_info)],
        )

        # Primal updates (exchange current estimates)
        admm_0.pose_primal_update(pg_0, {1: T_1})
        admm_1.pose_primal_update(pg_1, {0: T_0})

        T_0_new = admm_0.state.primal_pose
        T_1_new = admm_1.state.primal_pose

        # Dual update
        admm_0.dual_update({1: T_1_new})
        admm_1.dual_update({0: T_0_new})

        # Adaptive penalty
        r_01 = se3_log(torch.linalg.inv(T_0_new) @ T_1_new)
        primal_res = r_01.norm().item()
        dual_res = se3_log(torch.linalg.inv(T_0) @ T_0_new).norm().item()
        admm_0.update_penalty(primal_res, dual_res)
        admm_1.update_penalty(primal_res, dual_res)

        admm_0.state.iteration = it + 1
        admm_1.state.iteration = it + 1

        if admm_0.is_converged({1: T_1_new}) and admm_1.is_converged({0: T_0_new}):
            return True, it + 1, T_0_new, T_1_new

    return (
        False,
        max_iters,
        admm_0.state.primal_pose,
        admm_1.state.primal_pose,
    )


def test_admm_two_robots_converge():
    """2-robot synthetic pose graph converges in < 50 iterations."""
    torch.manual_seed(42)

    # Ground-truth shared pose
    xi_gt = torch.tensor([0.1, -0.05, 0.08, 0.5, -0.3, 0.2])
    T_gt = se3_exp(xi_gt)

    # Perturbed initial guesses (different perturbations per robot)
    noise_0 = se3_exp(torch.randn(6) * 0.2)
    noise_1 = se3_exp(torch.randn(6) * 0.3)
    T_init_0 = T_gt @ noise_0
    T_init_1 = T_gt @ noise_1

    converged, n_iters, _, _ = _run_admm_pair(
        T_gt, T_init_0, T_init_1, max_iters=50,
    )
    assert converged, f"ADMM did not converge in 50 iterations (ran {n_iters})"


def test_admm_preserves_gt():
    """Converged pose = known ground truth (synthetic)."""
    torch.manual_seed(77)

    xi_gt = torch.tensor([0.05, -0.03, 0.07, 0.3, -0.2, 0.1])
    T_gt = se3_exp(xi_gt)

    noise_0 = se3_exp(torch.randn(6) * 0.02)
    noise_1 = se3_exp(torch.randn(6) * 0.03)
    T_init_0 = T_gt @ noise_0
    T_init_1 = T_gt @ noise_1

    converged, _, T_final_0, T_final_1 = _run_admm_pair(
        T_gt, T_init_0, T_init_1, max_iters=50,
    )

    # Both converged poses should be close to the ground truth
    res_0 = se3_log(torch.linalg.inv(T_gt) @ T_final_0)
    res_1 = se3_log(torch.linalg.inv(T_gt) @ T_final_1)
    assert res_0.norm() < 1e-3, f"Robot 0 residual norm: {res_0.norm():.6f}"
    assert res_1.norm() < 1e-3, f"Robot 1 residual norm: {res_1.norm():.6f}"


def test_ablation_fim_faster():
    """FIM-weighted Gaussian consensus converges in fewer ADMM iterations than uniform."""
    torch.manual_seed(7)

    # Synthetic Gaussian parameters: 2 robots, 10 matched pairs
    N = 10
    gt_means = torch.randn(N, 3)

    # Robot 0: low noise (well-observed, high FIM)
    means_0 = gt_means + torch.randn(N, 3) * 0.01
    # Robot 1: high noise (poorly-observed, low FIM)
    means_1 = gt_means + torch.randn(N, 3) * 0.5

    matched_pairs = {1: (torch.arange(N), torch.arange(N))}

    # FIM-weighted: Robot 0 has high FIM, Robot 1 has low FIM
    local_fim_w  = {"means": torch.ones(N, 3) * 100.0}  # confident
    nb_fim_w     = {1: {"means": torch.ones(N, 3) * 0.1}}  # uncertain

    # Uniform: both have equal FIM
    local_fim_u  = {"means": torch.ones(N, 3)}
    nb_fim_u     = {1: {"means": torch.ones(N, 3)}}

    # Run a few ADMM-style iterations, measuring GT error after each
    admm_w = RiemannianADMM(robot_id=0, device="cpu")
    admm_w.rho = 1.0
    admm_u = RiemannianADMM(robot_id=0, device="cpu")
    admm_u.rho = 1.0

    iters_w, iters_u = 50, 50
    tol = 0.05  # mean L2 error to GT threshold

    # FIM-weighted run
    local_w = {"means": means_0.clone()}
    for i in range(50):
        local_w = admm_w.gaussian_primal_update(
            local_w, {1: {"means": means_1.clone()}},
            matched_pairs, local_fim_w, nb_fim_w,
        )
        err = (local_w["means"] - gt_means).norm(dim=-1).mean().item()
        if err < tol:
            iters_w = i + 1
            break

    # Uniform run
    local_u = {"means": means_0.clone()}
    for i in range(50):
        local_u = admm_u.gaussian_primal_update(
            local_u, {1: {"means": means_1.clone()}},
            matched_pairs, local_fim_u, nb_fim_u,
        )
        err = (local_u["means"] - gt_means).norm(dim=-1).mean().item()
        if err < tol:
            iters_u = i + 1
            break

    assert iters_w < iters_u, (
        f"FIM-weighted ({iters_w} iters) not faster than "
        f"uniform ({iters_u} iters)"
    )


def test_dual_update_shape():
    """After dual_update, dual_vars[nb].shape == (6,)."""
    admm = RiemannianADMM(robot_id=0, device="cpu")
    admm.initialize(torch.eye(4), neighbor_ids=[1, 2])

    T_1 = se3_exp(torch.tensor([0.1, 0.0, 0.0, 1.0, 0.0, 0.0]))
    T_2 = se3_exp(torch.tensor([0.0, 0.1, 0.0, 0.0, 1.0, 0.0]))

    admm.dual_update({1: T_1, 2: T_2})

    assert admm.state.dual_vars[1].shape == (6,)
    assert admm.state.dual_vars[2].shape == (6,)
    # Dual vars should be nonzero since neighbors differ from identity
    assert admm.state.dual_vars[1].norm() > 0
    assert admm.state.dual_vars[2].norm() > 0


def test_penalty_increases():
    """Penalty rho increases when primal_res >> dual_res."""
    admm = RiemannianADMM(robot_id=0, rho_init=1.0, tau_incr=2.0, mu=10.0, device="cpu")
    admm.initialize(torch.eye(4), neighbor_ids=[1])
    rho_before = admm.rho
    admm.update_penalty(primal_res=100.0, dual_res=1.0)
    assert admm.rho > rho_before


def test_converged_identical():
    """Two identical poses should be converged."""
    admm = RiemannianADMM(robot_id=0, device="cpu")
    T = torch.eye(4)
    admm.initialize(T, [1])
    assert admm.is_converged({1: T})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
