#!/bin/bash
# Run MAGiC-SLAM 2-agent for all remaining scenes (Apart-1, Apart-2, Office-0)
# Each scene takes ~1 hour for tracking. Total: ~3 hours.    
set -e

WORKSPACE="/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM"
PYTHON="/home/wen/anaconda3/envs/magic-slam/bin/python"
LOG_BASE="/tmp"

SCENES=("apart_1" "apart_2" "office_0")
SCENE_NAMES=("Apart-1" "Apart-2" "Office-0")

cd "$WORKSPACE"

for i in "${!SCENES[@]}"; do
    SCENE="${SCENES[$i]}"
    NAME="${SCENE_NAMES[$i]}"
    CONFIG="outputs/baselines/magic_slam/${SCENE}_2agent_single_gpu.yaml"
    LOG="$LOG_BASE/magic_${NAME}_2agent.log"
    
    echo ""
    echo "============================================================"
    echo "  Starting: $NAME (2-agent, patched max_threads=2)"
    echo "  Config: $CONFIG"
    echo "  Log: $LOG"
    echo "  Time: $(date)"
    echo "============================================================"
    
    $PYTHON scripts/run_magic_single_scene.py "$CONFIG" 2>&1 | tee "$LOG"
    
    echo "  Finished: $NAME at $(date)"
    echo ""
done

# Run recovery/eval on Apart-0 if needed (already done, skip)
echo ""
echo "============================================================"
echo "  ALL SCENES COMPLETE"
echo "  Results in: outputs/baselines/magic_slam/*-2agent/"
echo "============================================================"

# Collect results
echo ""
echo "=== ATE Summary ==="
for NAME in "Apart-0" "Apart-1" "Apart-2" "Office-0"; do
    DIR="outputs/baselines/magic_slam/${NAME}-2agent"
    if [ -f "$DIR/recovery_results.json" ]; then
        echo "$NAME: $(cat $DIR/recovery_results.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Agent0={d.get(\"agent_0_ate_aligned_cm\",\"?\"):.2f}cm, Agent1={d.get(\"agent_1_ate_aligned_cm\",\"?\"):.2f}cm, PSNR={d.get(\"fine_psnr\",\"?\"):.2f}")')"
    elif [ -d "$DIR" ]; then
        echo "$NAME: Run recovery script to get results"
    else
        echo "$NAME: Not yet run"
    fi
done
