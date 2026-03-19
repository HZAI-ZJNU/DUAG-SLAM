#!/usr/bin/env python3
"""scripts/extract_swarm.py — Inspect Swarm-SLAM for communication modules."""
import subprocess

result = subprocess.run(
    ["find", "repos/Swarm-SLAM", "-name", "*.py", "-not", "-path", "*/build/*"],
    capture_output=True, text=True
)
print("=== Swarm-SLAM Python files ===")
for f in sorted(result.stdout.strip().split("\n")):
    if f:
        print(f"  {f}")

print("\nsim_channel.py is already in extracted/swarm_comm/ (written from scratch).")
print("Real-robot path uses Swarm-SLAM ROS 2 directly — no extraction needed.")
