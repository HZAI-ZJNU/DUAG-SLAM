# core/types.py
#
# This file is the contract between all modules.
# Change carefully — a change to any shape here propagates everywhere.
#
# NOTE on SH storage (verified in repos/MonoGS/gaussian_splatting/scene/gaussian_model.py):
# MonoGS stores SH as two nn.Parameters:
#   _features_dc   [N, 1, 3]     degree-0 band
#   _features_rest [N, K-1, 3]   degrees 1..max_sh_degree, K=(max_sh_degree+1)^2
# We mirror this split. Do NOT merge into a single tensor.
#
# NOTE on quaternions (verified in repos/MonoGS/gaussian_splatting/utils/general_utils.py):
# MonoGS uses (w, x, y, z) ordering. Index 0 = w (scalar part).

import torch
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class GaussianMap:
    """
    One robot's local 3DGS map. All tensors on the same device.
    N = number of Gaussians. Shapes mirror MonoGS internal storage exactly.
    """
    # Geometry
    means:     torch.Tensor    # [N, 3]    raw 3D positions (_xyz)
    quats:     torch.Tensor    # [N, 4]    rotation quaternions (_rotation), (w,x,y,z)

    # Appearance
    scales:    torch.Tensor    # [N, 3]    log-space (_scaling); actual scale = exp(scales)
    opacities: torch.Tensor    # [N, 1]    logit-space (_opacity); actual = sigmoid(opacities)
    sh_dc:     torch.Tensor    # [N, 1, 3] degree-0 SH band (_features_dc)
    sh_rest:   torch.Tensor    # [N, K-1, 3] higher bands (_features_rest); K=(max_sh_degree+1)^2
                               #           For degree=3 (MonoGS default): [N, 15, 3]

    # Metadata
    robot_id:  int
    timestamp: float           # seconds since sequence start

    # Uncertainty (None until compute_gaussian_fim() has run)
    fim_means:  Optional[torch.Tensor] = None   # [N, 3]
    fim_quats:  Optional[torch.Tensor] = None   # [N, 4]
    fim_scales: Optional[torch.Tensor] = None   # [N, 3]
    fim_opac:   Optional[torch.Tensor] = None   # [N, 1]
    fim_sh_dc:  Optional[torch.Tensor] = None   # [N, 1, 3]


@dataclass
class PoseGraph:
    """
    Local pose graph for one robot.
    Edge format verified against repos/dpgo/include/DPGO/RelativeSEMeasurement.h.
    """
    poses: Dict[int, torch.Tensor]                          # node_id -> [4, 4] SE(3)
    edges: List[Tuple[int, int, torch.Tensor, torch.Tensor]]
    # edge = (node_i, node_j, relative_pose [4,4], information_matrix [6,6])
    robot_id: int


@dataclass
class ConsensusState:
    """Per-robot ADMM state. Lives on device."""
    primal_pose: torch.Tensor            # [4, 4] SE(3) current estimate
    dual_vars:   Dict[int, torch.Tensor] # neighbor_robot_id -> [6] in se(3)
    penalty:     float = 1.0            # rho (ADMM penalty)
    iteration:   int   = 0


@dataclass
class RobotMessage:
    """
    Atomic message between two robots.
    Simulation: Python objects passed directly via SimChannel.
    Real robots: serialized over ROS 2 topics via Swarm-SLAM infrastructure.
    """
    sender_id:   int
    receiver_id: int
    msg_type:    str       # "pose_update" | "gaussian_subset" | "loop_closure" | "admm_dual"
    payload:     bytes
    timestamp:   float
    priority:    float = 1.0   # CAIMUS sets this (Contribution 2); 0=drop, 1=send now
    byte_size:   int   = 0     # filled at send time for bandwidth tracking
