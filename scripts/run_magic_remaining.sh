#!/bin/bash
# Run MAGiC-SLAM on remaining Replica scenes sequentially
set -e
cd /home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/repos/MAGiC-SLAM
PYTHON=/home/wen/anaconda3/envs/magic-slam/bin/python
BASE=/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM/outputs/baselines/magic_slam

for scene in apart_1 apart_2 office_0; do
    CONFIG="${BASE}/${scene}_single_gpu.yaml"
    LOG="/tmp/magic_${scene}.log"
    echo "=== Starting ${scene} ==="
    echo "Config: ${CONFIG}"
    echo "Log: ${LOG}"
    $PYTHON run_slam.py "$CONFIG" > "$LOG" 2>&1
    echo "=== ${scene} DONE ==="
    tail -5 "$LOG"
    echo ""
done

echo "All scenes complete!"
