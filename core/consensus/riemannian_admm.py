# core/consensus/riemannian_admm.py
#
# Riemannian ADMM on SE(3) x G (product manifold).
#
# ARCHITECTURE:
#   Pose block  (SE(3)): one step via DPGOInterface or fallback Python gradient.
#   Gaussian block (G): geodesically convex solver (uncertainty_weighted_average).
#   Dual update       : runs in Python via se3_log.

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from core.types import ConsensusState, PoseGraph
from core.consensus.lie_algebra import se3_log, se3_exp


class RiemannianADMM:
    """
    Per-robot distributed ADMM optimizer on SE(3) x (S+3 x R^7)^N.

    One instance per robot. Robots exchange primal variables between iterations.
    The caller (RobotNode) is responsible for all inter-robot communication.
    """

    def __init__(
        self,
        robot_id:       int,
        dpgo_interface=None,     # extracted/dpgo_wrapper/DPGOInterface instance (optional)
        rho_init:       float = 1.0,
        rho_max:        float = 100.0,
        mu:             float = 10.0,
        tau_incr:       float = 2.0,
        tau_decr:       float = 2.0,
        tol_primal:     float = 1e-4,
        tol_dual:       float = 1e-4,
        lambda_gauss:   float = 1.0,
        pose_lr:        float = 0.1,
        device:         str   = "cuda",
    ):
        self.robot_id     = robot_id
        self.dpgo         = dpgo_interface
        self.rho          = rho_init
        self.rho_max      = rho_max
        self.mu           = mu
        self.tau_incr     = tau_incr
        self.tau_decr     = tau_decr
        self.tol_primal   = tol_primal
        self.tol_dual     = tol_dual
        self.lambda_gauss = lambda_gauss
        self.pose_lr      = pose_lr
        self.device       = device
        self.state: Optional[ConsensusState] = None

    def initialize(self, initial_pose: Tensor, neighbor_ids: List[int]) -> None:
        """
        Call once when first inter-robot loop closure is detected.
        """
        self.state = ConsensusState(
            primal_pose=initial_pose.to(self.device),
            dual_vars={nb: torch.zeros(6, device=self.device) for nb in neighbor_ids},
            penalty=self.rho,
        )

    def pose_primal_update(
        self,
        pose_graph:     PoseGraph,
        neighbor_poses: Dict[int, Tensor],
    ) -> Tensor:
        """
        SE(3) primal step.

        Computes a Riemannian gradient step minimizing the augmented Lagrangian
        on SE(3) for the pose consensus term:
            L(T_i) = sum_j [ <lambda_ij, Log(T_i^{-1} T_j)>
                           + (rho/2) ||Log(T_i^{-1} T_j)||^2 ]
        plus local pose-graph terms.

        Uses DPGO when available, falls back to pure-Python manifold gradient step.
        """
        assert self.state is not None
        T_i = self.state.primal_pose

        if self.dpgo is not None:
            return self._dpgo_pose_update(pose_graph, neighbor_poses)

        return self._python_pose_update(pose_graph, neighbor_poses)

    def _python_pose_update(
        self,
        pose_graph: PoseGraph,
        neighbor_poses: Dict[int, Tensor],
    ) -> Tensor:
        """
        Pure Python Riemannian gradient step on SE(3).

        Gradient of the augmented Lagrangian consensus term wrt T_i:
        For each neighbor j:
            r_ij = Log(T_i^{-1} T_j)     (residual in se(3))
            grad += -Adj(T_i)^T (lambda_ij + rho * r_ij)

        We also include local pose-graph edges that connect this robot's pose
        to itself (odometry constraints).

        Update: T_i <- T_i * exp(-lr * grad)
        """
        assert self.state is not None
        T_i = self.state.primal_pose
        grad = torch.zeros(6, device=self.device, dtype=T_i.dtype)

        # Consensus terms with neighbors
        for nb_id, T_j in neighbor_poses.items():
            T_ij = torch.linalg.inv(T_i) @ T_j
            r_ij = se3_log(T_ij)
            dual = self.state.dual_vars.get(nb_id, torch.zeros(6, device=self.device))
            # Gradient: -(dual + rho * residual)  (since we want to minimize)
            grad = grad - (dual + self.rho * r_ij)

        # Local pose-graph edges — use only the few most recent edges (not ALL
        # 2000+ odometry edges).  With many edges × high weight, odometry
        # overwhelms consensus and prevents inter-robot correction.
        # Also cap individual edge weight to be at most rho * 10.
        max_odom_weight = self.rho * 10.0
        max_odom_edges = 5
        edges = pose_graph.edges
        if len(edges) > max_odom_edges:
            edges = edges[-max_odom_edges:]
        for node_i, node_j, T_rel, info in edges:
            if node_i in pose_graph.poses and node_j in pose_graph.poses:
                T_a = pose_graph.poses[node_i]
                T_b = pose_graph.poses[node_j]
                T_pred = torch.linalg.inv(T_a) @ T_b
                r_edge = se3_log(torch.linalg.inv(T_rel) @ T_pred)
                weight = info.diag().mean() if info.dim() == 2 else info.mean()
                weight = min(weight.item() if isinstance(weight, Tensor) else weight,
                             max_odom_weight)
                grad = grad - weight * r_edge

        # Adaptive learning rate: scale inversely with total gradient magnitude
        # to maintain stability as rho and edge weights grow.
        effective_lr = self.pose_lr / (1.0 + grad.norm().item())

        # Retraction step on SE(3)
        step = se3_exp(-effective_lr * grad)
        new_pose = T_i @ step
        self.state.primal_pose = new_pose
        return new_pose

    def _dpgo_pose_update(
        self,
        pose_graph: PoseGraph,
        neighbor_poses: Dict[int, Tensor],
    ) -> Tensor:
        """
        SE(3) primal step via DPGO.
        """
        for nb_id, T_j in neighbor_poses.items():
            self.dpgo.set_neighbor_status(nb_id, self.state.iteration)

        self.dpgo.iterate()
        new_pose = self.dpgo.get_pose(0).to(self.device)
        self.state.primal_pose = new_pose
        return new_pose

    def gaussian_primal_update(
        self,
        local_params:     Dict[str, Tensor],
        neighbor_params:  Dict[int, Dict[str, Tensor]],
        matched_pairs:    Dict[int, Tuple[Tensor, Tensor]],
        local_fim:        Dict[str, Tensor],
        neighbor_fim:     Dict[int, Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Gaussian parameter primal step.

        Uses uncertainty_weighted_average with rho-augmented FIM weights.
        The proximal operator for the consensus + data-fidelity problem
        has a closed-form solution:
            effective_fim_a = fim_a + rho * ones_like(fim_a)
            effective_fim_b = fim_b + rho * ones_like(fim_b)
        """
        from core.uncertainty.propagation import uncertainty_weighted_average

        updated = {k: v.clone() for k, v in local_params.items()}

        for nb_id, nb_params in neighbor_params.items():
            if nb_id not in matched_pairs:
                continue

            local_idx, nb_idx = matched_pairs[nb_id]
            if len(local_idx) == 0:
                continue

            # Extract matched subsets
            params_a = {k: updated[k][local_idx] for k in updated}
            params_b = {k: nb_params[k][nb_idx] for k in nb_params if k in updated}

            nb_fim = neighbor_fim.get(nb_id, {})

            fim_a = {}
            fim_b = {}
            for k in params_a:
                fa = local_fim.get(k, torch.ones_like(params_a[k]))
                if isinstance(fa, Tensor) and fa.shape[0] > local_idx.max():
                    fa = fa[local_idx]
                fim_a[k] = fa + self.rho * torch.ones_like(fa)

                fb = nb_fim.get(k, torch.ones_like(params_b[k]))
                if isinstance(fb, Tensor) and fb.shape[0] > nb_idx.max():
                    fb = fb[nb_idx]
                fim_b[k] = fb + self.rho * torch.ones_like(fb)

            fused_params, _ = uncertainty_weighted_average(params_a, params_b, fim_a, fim_b)

            # Write back
            for k in fused_params:
                updated[k][local_idx] = fused_params[k]

        return updated

    def dual_update(self, neighbor_poses: Dict[int, Tensor]) -> None:
        """
        Lie algebra dual variable update.
        lambda_{ij} += rho * Log(T_i^{-1} @ T_j)
        """
        assert self.state is not None
        T_i = self.state.primal_pose
        for nb_id, T_j in neighbor_poses.items():
            if nb_id not in self.state.dual_vars:
                self.state.dual_vars[nb_id] = torch.zeros(6, device=T_i.device)
            residual = se3_log(torch.linalg.inv(T_i) @ T_j)
            self.state.dual_vars[nb_id] = (self.state.dual_vars[nb_id]
                                           + self.rho * residual)

    def update_penalty(self, primal_res: float, dual_res: float) -> None:
        """
        Adaptive rho update from Boyd et al. 2011, Section 3.4.1.
        """
        if primal_res > self.mu * dual_res:
            self.rho = min(self.rho * self.tau_incr, self.rho_max)
        elif dual_res > self.mu * primal_res:
            self.rho = max(self.rho / self.tau_decr, 1e-6)
        if self.state:
            self.state.penalty = self.rho

    def is_converged(self, neighbor_poses: Dict[int, Tensor]) -> bool:
        """
        True when all inter-robot pose residuals are below tol_primal.
        """
        if self.state is None:
            return False
        T_i = self.state.primal_pose
        for T_j in neighbor_poses.values():
            res = se3_log(torch.linalg.inv(T_i) @ T_j)
            if res.norm() > self.tol_primal:
                return False
        return True
