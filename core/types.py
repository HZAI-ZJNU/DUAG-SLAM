"""
DUAG-SLAM Core Data Types
==========================
Every module in the project communicates through these types.
This is the single source of truth for data structures.
"""
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np


@dataclass
class GaussianPrimitive:
    """
    A single 3D Gaussian primitive with uncertainty augmentation.
    
    Standard 3DGS attributes (from MonoGS):
        position:   μ ∈ R³        — center of the Gaussian
        covariance: Σ ∈ S⁺₃      — 3x3 symmetric positive definite (stored as 6 upper-tri)
        color:      c ∈ [0,1]³    — RGB color (spherical harmonics degree 0)
        opacity:    α ∈ [0,1]     — transparency
    
    OUR ADDITION (Sub-contribution 1.1):
        uncertainty: σ² ∈ R⁺      — epistemic uncertainty from Hessian (Eq. 5)
        observation_count: int    — how many frames observed this Gaussian
    """
    position: torch.Tensor       # shape: [3]
    covariance: torch.Tensor     # shape: [6] (upper triangular of 3x3)
    color: torch.Tensor          # shape: [3]
    opacity: torch.Tensor        # shape: [1]
    uncertainty: torch.Tensor    # shape: [1] — OUR NOVEL ADDITION
    observation_count: int = 0


@dataclass
class GaussianSubMap:
    """
    A collection of Gaussians representing one robot's local map.
    This is what Layer 1 produces and Layer 2/3 consume.
    """
    robot_id: int
    timestamp: float
    
    # All Gaussians stored as batched tensors for GPU efficiency
    positions: torch.Tensor       # shape: [N, 3]
    covariances: torch.Tensor     # shape: [N, 6]
    colors: torch.Tensor          # shape: [N, 3]
    opacities: torch.Tensor       # shape: [N, 1]
    uncertainties: torch.Tensor   # shape: [N, 1]  — OUR ADDITION
    
    # Associated pose graph
    keyframe_poses: torch.Tensor  # shape: [K, 4, 4] — SE(3) matrices
    keyframe_timestamps: List[float] = field(default_factory=list)
    
    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]
    
    @property
    def num_keyframes(self) -> int:
        return self.keyframe_poses.shape[0]
    
    def get_weight(self) -> torch.Tensor:
        """Consensus weight w_k = 1/σ_k² (Eq. 2). Higher confidence → higher weight."""
        return 1.0 / self.uncertainties.clamp(min=1e-8)


@dataclass
class GaussianAssociation:
    """
    Result of matching Gaussians between two robots' sub-maps.
    Produced by the association module, consumed by the DUAG-C optimizer.
    """
    robot_i: int
    robot_j: int
    indices_i: torch.Tensor       # shape: [M] — indices into robot i's sub-map
    indices_j: torch.Tensor       # shape: [M] — indices into robot j's sub-map
    match_confidence: torch.Tensor # shape: [M] — how confident each match is
    
    @property
    def num_matches(self) -> int:
        return self.indices_i.shape[0]


@dataclass 
class ConsensusResult:
    """Output of the DUAG-C optimizer."""
    relative_pose: torch.Tensor    # shape: [4, 4] — T_ij relating robot i to robot j
    merged_gaussians: GaussianSubMap  # The merged/refined sub-map
    converged: bool                # Did ADMM converge?
    iterations: int                # How many iterations were used
    primal_residual: float         # Final primal residual (convergence metric)
    pose_correction: torch.Tensor  # shape: [K, 4, 4] — corrections to local keyframes


@dataclass
class P2PMessage:
    """
    What actually gets transmitted between robots over the network.
    Layer 2 (CAIMUS) produces this from a GaussianSubMap.
    """
    sender_id: int
    receiver_id: int
    timestamp: float
    
    # Compressed Gaussian data (CAIMUS selects and compresses)
    compressed_positions: torch.Tensor   # shape: [M, 3] — M ≤ N (selected subset)
    compressed_covariances: torch.Tensor # shape: [M, 6]
    compressed_colors: torch.Tensor      # shape: [M, 3] — may be None in skeleton mode
    compressed_opacities: torch.Tensor   # shape: [M, 1]
    compressed_uncertainties: torch.Tensor # shape: [M, 1]
    
    # Place recognition descriptors (for TTA-PR)
    place_descriptors: torch.Tensor      # shape: [K, D] — per-keyframe descriptors
    
    # Optional: LoRA adapter weights (for TTA-PR cross-robot distillation)
    adapter_weights: Optional[Dict[str, torch.Tensor]] = None
    
    @property
    def size_bytes(self) -> int:
        """Estimate the message size for bandwidth scheduling."""
        total = 0
        for attr in [self.compressed_positions, self.compressed_covariances,
                     self.compressed_colors, self.compressed_opacities,
                     self.compressed_uncertainties, self.place_descriptors]:
            if attr is not None:
                total += attr.numel() * attr.element_size()
        if self.adapter_weights:
            for v in self.adapter_weights.values():
                total += v.numel() * v.element_size()
        return total
