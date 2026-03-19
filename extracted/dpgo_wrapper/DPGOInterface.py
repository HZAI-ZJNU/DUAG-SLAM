# extracted/dpgo_wrapper/DPGOInterface.py
# Python wrapper around the compiled dpgo_pybind module.
# Called by core/consensus/riemannian_admm.py:RiemannianADMM.pose_primal_update().

import sys
import os
import torch
import numpy as np

# Add build directory to path for dpgo_pybind import
_build_dir = os.path.join(os.path.dirname(__file__), "build")
if _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)


class DPGOInterface:
    """
    Translates between core/types.py PoseGraph and DPGO's C++ internal format.
    One instance per robot. Stateful — wraps a single C++ PGOAgent object.
    """

    def __init__(self, robot_id: int, n_robots: int):
        try:
            import dpgo_pybind
        except ImportError:
            raise RuntimeError(
                "dpgo_pybind not found. Build it with: "
                "cd extracted/dpgo_wrapper && mkdir -p build && cd build && "
                "cmake .. && make -j$(nproc)"
            )
        # d=3 (3D), r=3 (relaxation rank = dimension for SE(3))
        params = dpgo_pybind.PGOAgentParameters(d=3, r=3, num_robots=n_robots)
        params.verbose = False
        self._agent = dpgo_pybind.PGOAgent(robot_id, params)
        self.robot_id = robot_id
        self._initialized = False

    def add_odometry(
        self,
        node_i: int,
        node_j: int,
        T_ij: torch.Tensor,    # [4, 4] SE(3) relative pose
        Omega: torch.Tensor,   # [6, 6] information matrix
    ) -> None:
        """Add a local odometry edge between consecutive poses."""
        import dpgo_pybind
        R = T_ij[:3, :3].detach().cpu().numpy().astype(np.float64)
        t = T_ij[:3, 3].detach().cpu().numpy().astype(np.float64)
        # Use mean of rotational and translational diagonal blocks as precision
        kappa = float(Omega[:3, :3].diag().mean())
        tau = float(Omega[3:, 3:].diag().mean())
        meas = dpgo_pybind.RelativeSEMeasurement(
            self.robot_id, self.robot_id, node_i, node_j, R, t, kappa, tau
        )
        self._agent.addMeasurement(meas)

    def add_loop_closure(
        self,
        robot_i: int, node_i: int,
        robot_j: int, node_j: int,
        T_ij: torch.Tensor,    # [4, 4] SE(3) inter-robot relative pose
        Omega: torch.Tensor,   # [6, 6] information matrix
    ) -> None:
        """Add an inter-robot loop closure edge."""
        import dpgo_pybind
        R = T_ij[:3, :3].detach().cpu().numpy().astype(np.float64)
        t = T_ij[:3, 3].detach().cpu().numpy().astype(np.float64)
        kappa = float(Omega[:3, :3].diag().mean())
        tau = float(Omega[3:, 3:].diag().mean())
        meas = dpgo_pybind.RelativeSEMeasurement(
            robot_i, robot_j, node_i, node_j, R, t, kappa, tau
        )
        self._agent.addMeasurement(meas)

    def initialize(self) -> None:
        """Initialize the agent (must be called after adding measurements)."""
        self._agent.initialize()
        self._initialized = True

    def set_neighbor_status(self, neighbor_id: int, iteration: int) -> None:
        """Update neighbor's status for ADMM synchronization."""
        import dpgo_pybind
        status = dpgo_pybind.PGOAgentStatus()
        status.agentID = neighbor_id
        status.iterationNumber = iteration
        self._agent.setNeighborStatus(status)

    def iterate(self, do_optimization: bool = True) -> bool:
        """Run one Riemannian gradient step. Returns True if successful."""
        if not self._initialized:
            raise RuntimeError("Call initialize() before iterate()")
        return self._agent.iterate(do_optimization)

    def get_pose(self, node_id: int) -> torch.Tensor:
        """
        Returns this robot's current pose estimate as [4, 4] SE(3) float tensor.
        Converts from d x (d+1) Eigen matrix returned by DPGO to 4x4 homogeneous.
        """
        M = self._agent.getPoseInGlobalFrame(node_id)  # d x (d+1) = 3 x 4
        T = torch.eye(4, dtype=torch.float64)
        T[:3, :] = torch.from_numpy(M.astype(np.float64))
        return T.float()

    def get_trajectory_local(self) -> torch.Tensor:
        """Returns the full trajectory in local frame as [n, 4, 4] tensor."""
        traj = self._agent.getTrajectoryInLocalFrame()  # d x (d+1)*n
        n = traj.shape[1] // 4  # (d+1) = 4 for 3D
        poses = []
        for i in range(n):
            T = torch.eye(4, dtype=torch.float64)
            T[:3, :] = torch.from_numpy(traj[:, i*4:(i+1)*4].astype(np.float64))
            poses.append(T)
        return torch.stack(poses).float()

    @property
    def num_poses(self) -> int:
        return self._agent.num_poses()

    def reset(self) -> None:
        """Reset the agent for a new optimization round."""
        self._agent.reset()
        self._initialized = False
