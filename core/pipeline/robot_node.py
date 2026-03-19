# core/pipeline/robot_node.py
#
# Per-robot DUAG-SLAM pipeline orchestrator.
# One instance per robot. Processes RGB-D frames and participates in
# decentralized consensus with neighbors.

import torch
from torch import Tensor
from typing import Dict, List, Optional, Callable
from core.types import GaussianMap, PoseGraph, RobotMessage, ConsensusState
from core.consensus.riemannian_admm import RiemannianADMM
from core.consensus.gaussian_consensus import GaussianConsensus
from core.consensus.convergence import ConvergenceMonitor
from core.consensus.matching import match_gaussians
from core.consensus.lie_algebra import se3_log


class RobotNode:
    """
    One instance per robot. Runs the full DUAG-SLAM pipeline for that robot.

    State:
        robot_id:            int
        gaussian_map:        GaussianMap  (local 3DGS map)
        pose_graph:          PoseGraph    (local odometry + inter-robot edges)
        admm:                RiemannianADMM  (None until first loop closure)
        gauss_consensus:     GaussianConsensus
        convergence_monitor: ConvergenceMonitor
        comm:                SimChannel-like object with send()/receive() methods
        slam_step:           Callable  (local SLAM update, injected for testability)

    Per-frame call sequence (call process_frame() once per RGB-D frame):
        1. slam_step(rgb, depth, timestamp) -> updates gaussian_map, current_pose
        2. Every K_fim frames: recompute FIM weights
        3. If ADMM is active: run consensus round
    """

    def __init__(
        self,
        robot_id: int,
        config: dict,
        comm_channel,
        slam_step: Optional[Callable] = None,
        dpgo_interface=None,
    ):
        self.robot_id = robot_id
        self.config = config
        self.comm = comm_channel

        # Local SLAM callback: (rgb, depth, timestamp) -> (GaussianMap, Tensor[4,4])
        # If None, a no-op is used (for smoke testing without MonoGS).
        self._slam_step = slam_step

        device = config.get("device", "cpu")
        self.device = device

        # Initialize empty map
        N_init = config.get("n_init_gaussians", 100)
        self.gaussian_map = GaussianMap(
            means=torch.zeros(N_init, 3, device=device),
            quats=torch.tensor([[1.0, 0, 0, 0]], device=device).expand(N_init, -1).clone(),
            scales=torch.zeros(N_init, 3, device=device),
            opacities=torch.zeros(N_init, 1, device=device),
            sh_dc=torch.zeros(N_init, 1, 3, device=device),
            sh_rest=torch.zeros(N_init, 15, 3, device=device),
            robot_id=robot_id,
            timestamp=0.0,
        )

        # Current pose
        self.current_pose = torch.eye(4, device=device)

        # Pose graph
        self.pose_graph = PoseGraph(
            poses={0: torch.eye(4, device=device)},
            edges=[],
            robot_id=robot_id,
        )

        # Consensus modules (lazy init for ADMM)
        self.admm: Optional[RiemannianADMM] = None
        self.gauss_consensus = GaussianConsensus(
            robot_id=robot_id,
            match_distance_thresh=config.get("match_distance_thresh", 0.05),
            opacity_thresh=config.get("match_opacity_thresh", 0.1),
            device=device,
        )
        self.convergence_monitor = ConvergenceMonitor(
            window_size=config.get("convergence_window", 10),
        )
        self.dpgo_interface = dpgo_interface

        # Tracking
        self.frame_count = 0
        self.k_fim = config.get("k_fim", 10)
        self.admm_config = {
            "rho_init": config.get("rho_init", 1.0),
            "rho_max": config.get("rho_max", 100.0),
            "pose_lr": config.get("pose_lr", 0.1),
            "tol_primal": config.get("tol_primal", 1e-4),
            "tol_dual": config.get("tol_dual", 1e-4),
            "device": device,
        }

        # Camera intrinsics for FIM visibility check
        if "fx" in config:
            self._camera_intrinsics = (
                config["fx"], config["fy"], config["cx"], config["cy"]
            )
            self._image_size = (config.get("H", 680), config.get("W", 1200))

        # Neighbor data cache
        self._neighbor_maps: Dict[int, GaussianMap] = {}
        self._neighbor_poses: Dict[int, Tensor] = {}
        self.use_fim_weighting = config.get("use_fim_weighting", True)

    def process_frame(self, rgb: Tensor, depth: Tensor, timestamp: float) -> None:
        """
        Main per-frame entry point.

        1. Run local SLAM step
        2. Periodically recompute FIM
        3. Process incoming messages
        4. Run ADMM consensus if active
        """
        self.frame_count += 1

        # 1. Local SLAM update
        if self._slam_step is not None:
            result = self._slam_step(rgb, depth, timestamp)
            if result is not None:
                self.gaussian_map, self.current_pose = result
        else:
            # Synthetic mode: simulate pose drift and map update
            self._synthetic_slam_step(rgb, depth, timestamp)

        # Update pose graph with new pose
        node_id = self.frame_count
        self.pose_graph.poses[node_id] = self.current_pose.clone()
        if node_id > 1:
            # Add odometry edge
            T_prev = self.pose_graph.poses[node_id - 1]
            T_rel = torch.linalg.inv(T_prev) @ self.current_pose
            info = torch.eye(6, device=self.device) * 100.0  # high-confidence odometry
            self.pose_graph.edges.append((node_id - 1, node_id, T_rel, info))

        self.gaussian_map.timestamp = timestamp

        # FIM weights: use gradient-based FIM from SLAM if available,
        # otherwise use view-count-based heuristic.
        if self.use_fim_weighting and self.frame_count % self.k_fim == 0:
            self._compute_synthetic_fim()

        # 2. Process incoming messages
        self._process_messages()

        # 3. ADMM consensus round (if active)
        if self.admm is not None and self.admm.state is not None:
            self._consensus_round()

    def add_loop_closure(self, neighbor_id: int, relative_pose: Tensor, info: Tensor) -> None:
        """
        Called when an inter-robot loop closure is detected.
        Initializes ADMM if not already running.
        """
        # Add edge to pose graph
        local_node = max(self.pose_graph.poses.keys())
        self.pose_graph.edges.append((local_node, -(neighbor_id + 1), relative_pose, info))

        if self.admm is None:
            self.admm = RiemannianADMM(
                robot_id=self.robot_id,
                dpgo_interface=self.dpgo_interface,
                **self.admm_config,
            )
            self.admm.initialize(self.current_pose, neighbor_ids=[neighbor_id])
        elif neighbor_id not in self.admm.state.dual_vars:
            self.admm.state.dual_vars[neighbor_id] = torch.zeros(6, device=self.device)

    def _consensus_round(self) -> None:
        """Run one ADMM iteration with available neighbor data."""
        if not self._neighbor_poses:
            return

        # Pose primal update
        prev_pose = self.admm.state.primal_pose.clone()
        new_pose = self.admm.pose_primal_update(self.pose_graph, self._neighbor_poses)

        # Gaussian primal update (if we have neighbor maps)
        if self._neighbor_maps:
            local_params = {
                "means": self.gaussian_map.means,
                "quats": self.gaussian_map.quats,
                "scales": self.gaussian_map.scales,
                "opacities": self.gaussian_map.opacities,
                "sh_dc": self.gaussian_map.sh_dc,
            }
            local_fim = {
                "means": self.gaussian_map.fim_means if self.gaussian_map.fim_means is not None
                         else torch.ones_like(self.gaussian_map.means),
                "quats": self.gaussian_map.fim_quats if self.gaussian_map.fim_quats is not None
                         else torch.ones_like(self.gaussian_map.quats),
                "scales": self.gaussian_map.fim_scales if self.gaussian_map.fim_scales is not None
                          else torch.ones_like(self.gaussian_map.scales),
                "opacities": self.gaussian_map.fim_opac if self.gaussian_map.fim_opac is not None
                             else torch.ones_like(self.gaussian_map.opacities),
                "sh_dc": self.gaussian_map.fim_sh_dc if self.gaussian_map.fim_sh_dc is not None
                         else torch.ones_like(self.gaussian_map.sh_dc),
            }

            neighbor_params = {}
            neighbor_fim = {}
            matched_pairs = {}
            for nb_id, nb_map in self._neighbor_maps.items():
                m_a, m_b, _, _ = match_gaussians(
                    self.gaussian_map.means, nb_map.means,
                    self.gaussian_map.opacities, nb_map.opacities,
                    distance_thresh=self.gauss_consensus.match_distance_thresh,
                    opacity_thresh=self.gauss_consensus.opacity_thresh,
                )
                if len(m_a) > 0:
                    neighbor_params[nb_id] = {
                        "means": nb_map.means,
                        "quats": nb_map.quats,
                        "scales": nb_map.scales,
                        "opacities": nb_map.opacities,
                        "sh_dc": nb_map.sh_dc,
                    }
                    neighbor_fim[nb_id] = {
                        "means": nb_map.fim_means if nb_map.fim_means is not None
                                 else torch.ones_like(nb_map.means),
                        "quats": nb_map.fim_quats if nb_map.fim_quats is not None
                                 else torch.ones_like(nb_map.quats),
                        "scales": nb_map.fim_scales if nb_map.fim_scales is not None
                                  else torch.ones_like(nb_map.scales),
                        "opacities": nb_map.fim_opac if nb_map.fim_opac is not None
                                     else torch.ones_like(nb_map.opacities),
                        "sh_dc": nb_map.fim_sh_dc if nb_map.fim_sh_dc is not None
                                 else torch.ones_like(nb_map.sh_dc),
                    }
                    matched_pairs[nb_id] = (m_a, m_b)

            if neighbor_params:
                updated = self.admm.gaussian_primal_update(
                    local_params, neighbor_params, matched_pairs, local_fim, neighbor_fim,
                )
                self.gaussian_map.means = updated["means"]
                self.gaussian_map.quats = updated["quats"]
                self.gaussian_map.scales = updated["scales"]
                self.gaussian_map.opacities = updated["opacities"]
                self.gaussian_map.sh_dc = updated["sh_dc"]

        # Dual update
        self.admm.dual_update(self._neighbor_poses)

        # Compute residuals
        T_i = self.admm.state.primal_pose
        primal_res = max(
            se3_log(torch.linalg.inv(T_i) @ T_j).norm().item()
            for T_j in self._neighbor_poses.values()
        )
        dual_res = se3_log(torch.linalg.inv(prev_pose) @ T_i).norm().item()

        # Update penalty and monitor
        self.admm.update_penalty(primal_res, dual_res)
        self.convergence_monitor.update(primal_res, dual_res, self.admm.rho)
        self.admm.state.iteration += 1

        # Send updated pose to neighbors
        self._broadcast_pose()

        # Check convergence
        if self.admm.is_converged(self._neighbor_poses):
            self._finalize_consensus()

    def _finalize_consensus(self) -> None:
        """Fuse neighbor maps after convergence and reset ADMM."""
        for nb_id, nb_map in self._neighbor_maps.items():
            T_i = self.admm.state.primal_pose
            T_j = self._neighbor_poses.get(nb_id, T_i)
            T_i_from_j = torch.linalg.inv(T_i) @ T_j
            nb_transformed = self.gauss_consensus.transform_gaussians(nb_map, T_i_from_j)
            self.gaussian_map = self.gauss_consensus.fuse(self.gaussian_map, nb_transformed)

        self.admm = None
        self._neighbor_maps.clear()
        self._neighbor_poses.clear()

    def _process_messages(self) -> None:
        """Process incoming messages from the communication channel."""
        messages = self.comm.receive(self.robot_id)
        for msg in messages:
            if msg.msg_type == "pose_update":
                pose = torch.frombuffer(bytearray(msg.payload), dtype=torch.float32).reshape(4, 4)
                self._neighbor_poses[msg.sender_id] = pose.to(self.device)
            elif msg.msg_type == "loop_closure":
                # payload: 4x4 relative pose + 6x6 information matrix
                data = torch.frombuffer(bytearray(msg.payload), dtype=torch.float32)
                rel_pose = data[:16].reshape(4, 4).to(self.device)
                info = data[16:52].reshape(6, 6).to(self.device)
                self.add_loop_closure(msg.sender_id, rel_pose, info)

    def _broadcast_pose(self) -> None:
        """Send current pose to all known neighbors."""
        if self.admm is None or self.admm.state is None:
            return
        pose_bytes = self.admm.state.primal_pose.cpu().float().numpy().tobytes()
        for nb_id in self.admm.state.dual_vars:
            msg = RobotMessage(
                sender_id=self.robot_id,
                receiver_id=nb_id,
                msg_type="pose_update",
                payload=pose_bytes,
                timestamp=self.gaussian_map.timestamp,
                byte_size=len(pose_bytes),
            )
            self.comm.send(msg)

    def _synthetic_slam_step(self, rgb: Tensor, depth: Tensor, timestamp: float) -> None:
        """
        Synthetic SLAM step for smoke testing without MonoGS.
        Simulates small pose drift and random Gaussian map updates.
        """
        # Small forward translation per frame
        delta = torch.eye(4, device=self.device)
        delta[2, 3] = 0.01  # 1cm forward
        self.current_pose = self.current_pose @ delta

        # Add slight noise to some Gaussian means
        N = self.gaussian_map.means.shape[0]
        noise = torch.randn(N, 3, device=self.device) * 0.001
        self.gaussian_map.means = self.gaussian_map.means + noise

    def _compute_synthetic_fim(self) -> None:
        """
        Compute FIM weights based on distance from current camera position.
        Closer Gaussians are better observed → higher FIM.
        Overwrites each call (snapshot of current observability).
        """
        cam_pos = self.current_pose[:3, 3]
        dists = (self.gaussian_map.means - cam_pos.unsqueeze(0)).norm(dim=1, keepdim=True)
        fim_weight = 1.0 / (1.0 + dists ** 2)

        self.gaussian_map.fim_means = fim_weight.expand(-1, 3)
        self.gaussian_map.fim_quats = fim_weight.expand(-1, 4)
        self.gaussian_map.fim_scales = fim_weight.expand(-1, 3)
        self.gaussian_map.fim_opac = fim_weight
        self.gaussian_map.fim_sh_dc = fim_weight.unsqueeze(-1).expand(-1, 1, 3)
