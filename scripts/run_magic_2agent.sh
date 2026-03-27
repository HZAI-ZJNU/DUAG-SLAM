#!/bin/bash
# Run MAGiC-SLAM for all scenes with 2 agents on single GPU
set -e
PYTHON=/home/wen/anaconda3/envs/magic-slam/bin/python
MAGIC_DIR=/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/repos/MAGiC-SLAM
CONFIG_DIR=/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/outputs/baselines/magic_slam

cd "$MAGIC_DIR"

echo "=== Starting 2-agent MAGiC-SLAM runs ==="

for scene in apart_0 apart_1 apart_2 office_0; do
    CONFIG="$CONFIG_DIR/${scene}_2agent_single_gpu.yaml"
    LOG="/tmp/magic_${scene}_2agent.log"
    echo "$(date): Running $scene with 2 agents..."
    $PYTHON run_slam.py "$CONFIG" > "$LOG" 2>&1
    echo "$(date): $scene done. Log: $LOG"
done

echo "=== All 2-agent runs complete ==="
