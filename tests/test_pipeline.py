# tests/test_pipeline.py
#
# STEP 12 — Full end-to-end smoke test for RobotNode pipeline.

import torch
from collections import defaultdict
from core.types import RobotMessage, GaussianMap
from core.pipeline.robot_node import RobotNode
from extracted.swarm_comm.sim_channel import SimChannel


class SimpleChannel:
    """Minimal SimChannel for testing — no bandwidth/loss simulation."""
    def __init__(self):
        self._queues = defaultdict(list)

    def send(self, msg: RobotMessage) -> bool:
        self._queues[msg.receiver_id].append(msg)
        return True

    def receive(self, robot_id: int):
        msgs = self._queues[robot_id]
        self._queues[robot_id] = []
        return msgs


def test_smoke_two_robots():
    """Two RobotNodes process 10 frames without crash."""
    channel = SimpleChannel()
    config = {
        "device": "cpu",
        "n_init_gaussians": 50,
        "k_fim": 5,
        "match_distance_thresh": 0.1,
        "rho_init": 1.0,
        "pose_lr": 0.05,
        "tol_primal": 1e-3,
    }

    robot_a = RobotNode(robot_id=0, config=config, comm_channel=channel)
    robot_b = RobotNode(robot_id=1, config=config, comm_channel=channel)

    # Process 10 frames each
    for t in range(10):
        rgb = torch.randn(480, 640, 3)
        depth = torch.rand(480, 640)
        ts = float(t) * 0.033

        robot_a.process_frame(rgb, depth, ts)
        robot_b.process_frame(rgb, depth, ts)

    # Verify both robots advanced their pose and frame count
    assert robot_a.frame_count == 10
    assert robot_b.frame_count == 10
    assert robot_a.current_pose.shape == (4, 4)
    assert robot_b.current_pose.shape == (4, 4)

    # Verify pose graphs have edges
    assert len(robot_a.pose_graph.edges) == 9  # 10 frames -> 9 odometry edges
    assert len(robot_b.pose_graph.edges) == 9


def test_smoke_with_loop_closure():
    """Two robots with a loop closure trigger ADMM without crash."""
    channel = SimpleChannel()
    config = {
        "device": "cpu",
        "n_init_gaussians": 30,
        "k_fim": 5,
        "match_distance_thresh": 1.0,  # generous threshold for synthetic data
        "rho_init": 1.0,
        "pose_lr": 0.05,
        "tol_primal": 0.5,  # loose tolerance for quick convergence in test
    }

    robot_a = RobotNode(robot_id=0, config=config, comm_channel=channel)
    robot_b = RobotNode(robot_id=1, config=config, comm_channel=channel)

    # Process a few frames
    for t in range(3):
        rgb = torch.randn(480, 640, 3)
        depth = torch.rand(480, 640)
        robot_a.process_frame(rgb, depth, float(t))
        robot_b.process_frame(rgb, depth, float(t))

    # Trigger loop closure between the two robots
    rel_pose = torch.eye(4)
    rel_pose[:3, 3] = torch.tensor([0.01, 0.0, 0.0])  # small relative offset
    info = torch.eye(6) * 10.0

    robot_a.add_loop_closure(neighbor_id=1, relative_pose=rel_pose, info=info)
    assert robot_a.admm is not None

    # Inject neighbor pose so ADMM can run
    robot_a._neighbor_poses[1] = robot_b.current_pose.clone()

    # Process more frames — ADMM should run without crash
    for t in range(3, 8):
        rgb = torch.randn(480, 640, 3)
        depth = torch.rand(480, 640)
        robot_a.process_frame(rgb, depth, float(t))
        robot_b.process_frame(rgb, depth, float(t))

    assert robot_a.frame_count == 8
    assert robot_b.frame_count == 8


def test_send_receive():
    """SimChannel delivers messages correctly."""
    ch = SimChannel()
    msg = RobotMessage(0, 1, "pose_update", b"hello", 0., byte_size=5)
    assert ch.send(msg)
    msgs = ch.receive(1)
    assert len(msgs) == 1 and msgs[0].payload == b"hello"


def test_empty_receive():
    """Receiving from empty queue returns empty list."""
    assert SimChannel().receive(99) == []


def test_loss_rate_one():
    """100% loss rate drops all messages."""
    ch = SimChannel(loss_rate=1.0)
    for _ in range(20):
        ch.send(RobotMessage(0, 1, "pose_update", b"x", 0., byte_size=1))
    assert ch.receive(1) == []


def test_bandwidth_throttle():
    """Messages exceeding bandwidth limit are dropped."""
    ch = SimChannel(bandwidth_bps=100)
    ch.send(RobotMessage(0, 1, "pose_update", b"x" * 100, 0., byte_size=100))
    dropped = not ch.send(RobotMessage(0, 1, "pose_update", b"y" * 200, 0., byte_size=200))
    assert dropped
