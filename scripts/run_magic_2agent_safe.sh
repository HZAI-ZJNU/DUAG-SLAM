#!/bin/bash
# Run MAGiC-SLAM 2-agent for all scenes with patched thread count.
# For scenes with existing submaps (Apart-0), uses recovery mode.
# For scenes without, runs full pipeline with monkey-patched max_threads.
set -e

WORKSPACE="/home/wen/Desktop/Project-Slam/C1-DAUG/DUAG-SLAM"
PYTHON="/home/wen/anaconda3/envs/magic-slam/bin/python"
MAGIC_DIR="$WORKSPACE/repos/MAGiC-SLAM"

SCENES=("Apart-0" "Apart-1" "Apart-2" "Office-0")

cd "$MAGIC_DIR"

for SCENE in "${SCENES[@]}"; do
    OUTPUT_DIR="$WORKSPACE/outputs/baselines/magic_slam/${SCENE}-2agent"
    SUBMAP_DIR="$OUTPUT_DIR/agent_0/submaps"
    CONFIG="$WORKSPACE/outputs/baselines/magic_slam/$(echo $SCENE | tr '[:upper:]' '[:lower:]' | tr '-' '_')_2agent_single_gpu.yaml"
    
    # Check if submaps already exist (recovery mode)
    if [ -d "$SUBMAP_DIR" ] && [ "$(ls -1 $SUBMAP_DIR/*.ckpt 2>/dev/null | wc -l)" -gt 0 ]; then
        echo ""
        echo "=== $SCENE: Submaps exist — using RECOVERY mode ==="
        $PYTHON "$WORKSPACE/scripts/magic_recover_and_eval.py" \
            --scene "$SCENE" --do_pgo --max_threads 2 \
            2>&1 | tee "/tmp/magic_${SCENE}_recovery.log"
    else
        echo ""
        echo "=== $SCENE: No submaps — running FULL pipeline ==="
        echo "Config: $CONFIG"
        
        if [ ! -f "$CONFIG" ]; then
            echo "ERROR: Config not found: $CONFIG"
            continue
        fi
        
        # Run with monkey-patched max_threads via a wrapper
        $PYTHON -c "
import sys, os
os.chdir('$MAGIC_DIR')
sys.path.insert(0, '.')

# Monkey-patch register_agents_submaps to use max_threads=2
import src.entities.magic_slam as ms
_orig_optimize = ms.MAGiCSLAM.optimize_poses

def _patched_optimize(self, agents_submaps):
    from src.entities.loop_detection.loop_detector import LoopDetector
    from src.entities.pose_graph_adapter import PoseGraphAdapter
    from src.utils.magic_slam_utils import register_agents_submaps, register_submaps, apply_pose_correction
    import numpy as np
    
    loop_detector = LoopDetector(self.config['loop_detection'])
    agents_c2ws = {}
    for agent_id, agent_submaps in agents_submaps.items():
        agents_c2ws[agent_id] = np.vstack([s['submap_c2ws'] for s in agent_submaps])
    
    intra_loops, inter_loops = loop_detector.detect_loops(agents_submaps)
    print(f'Detected {len(intra_loops)} intra + {len(inter_loops)} inter loops')
    
    # KEY FIX: max_threads=2 instead of 20
    intra_loops = register_agents_submaps(agents_submaps, intra_loops, register_submaps, max_threads=2)
    inter_loops = register_agents_submaps(agents_submaps, inter_loops, register_submaps, max_threads=2)
    
    intra_loops = loop_detector.filter_loops(intra_loops)
    inter_loops = loop_detector.filter_loops(inter_loops)
    print(f'After filtering: {len(intra_loops)} intra + {len(inter_loops)} inter loops')
    
    graph_wrapper = PoseGraphAdapter(agents_submaps, intra_loops + inter_loops)
    if len(intra_loops + inter_loops) > 0:
        graph_wrapper.optimize()
    
    return apply_pose_correction(graph_wrapper.get_poses(), agents_submaps)

ms.MAGiCSLAM.optimize_poses = _patched_optimize
print('Patched optimize_poses: max_threads=2')

from run_slam import main
sys.argv = ['run_slam.py', '$CONFIG']
main()
" 2>&1 | tee "/tmp/magic_${SCENE}_2agent.log"
    fi
    
    echo "=== $SCENE done ==="
done

echo ""
echo "=== All scenes complete ==="
echo "Results in: $WORKSPACE/outputs/baselines/magic_slam/*-2agent/"
