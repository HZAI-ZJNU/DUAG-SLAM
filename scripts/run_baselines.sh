#!/usr/bin/env bash
# scripts/run_baselines.sh — Run MAGiC-SLAM and MAC-Ego3D baselines on ReplicaMultiagent
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT}/data/ReplicaMultiagent"
OUTPUT_BASE="${ROOT}/outputs/baselines"

SCENES=("Apart-0" "Apart-1" "Apart-2" "Office-0")
SCENE_CONFIGS=("apart_0" "apart_1" "apart_2" "office_0")
SCENE_PARTS_1=("apart_0_part1" "apart_1_part1" "apart_2_part1" "office_0_part1")
SCENE_PARTS_2=("apart_0_part2" "apart_1_part2" "apart_2_part2" "office_0_part2")

# Check dataset exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Dataset not found at ${DATA_DIR}"
    echo "Download it first (Step 13)."
    exit 1
fi

# ============================================================
# MAGiC-SLAM baseline
# NOTE: Requires n+1 GPUs (3 for 2 agents). With 1 GPU, runs
# single-agent mode (agent_ids=[0], multi_gpu=False).
# ============================================================
echo "=========================================="
echo " MAGiC-SLAM Baseline"
echo "=========================================="

MAGIC_DIR="${ROOT}/repos/MAGiC-SLAM"
MAGIC_OUT="${OUTPUT_BASE}/magic_slam"

if [ -f "${MAGIC_DIR}/run_slam.py" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
    echo "Detected ${NUM_GPUS} GPU(s)"

    for i in "${!SCENES[@]}"; do
        scene="${SCENES[$i]}"
        cfg="${SCENE_CONFIGS[$i]}"
        out_dir="${MAGIC_OUT}/${scene}"
        mkdir -p "${out_dir}"

        echo "--- MAGiC-SLAM: ${scene} ---"

        # Create single-GPU override config if needed
        if [ "${NUM_GPUS}" -lt 3 ]; then
            echo "WARNING: MAGiC-SLAM needs 3 GPUs for 2-agent mode, have ${NUM_GPUS}."
            echo "Running single-agent mode (agent 0 only)."
            override_cfg="${MAGIC_OUT}/${cfg}_single_gpu.yaml"
            cat > "${override_cfg}" <<EOF
inherit_from: configs/ReplicaMultiagent/${cfg}.yaml
multi_gpu: False
data:
  agent_ids: [0]
  input_path: ${DATA_DIR}/${scene}/
  output_path: ${out_dir}/
EOF
            cfg_path="${override_cfg}"
        else
            cfg_path="configs/ReplicaMultiagent/${cfg}.yaml"
        fi

        (
            cd "${MAGIC_DIR}"
            PYTHONPATH="" python run_slam.py "${cfg_path}" \
                --input_path "${DATA_DIR}/${scene}/" \
                --output_path "${out_dir}/" \
                2>&1 | tee "${out_dir}/run.log"
        ) || echo "WARNING: MAGiC-SLAM failed on ${scene} (exit $?). Continuing..."
    done
else
    echo "SKIP: MAGiC-SLAM not found at ${MAGIC_DIR}"
fi

# ============================================================
# MAC-Ego3D baseline
# Runs on single GPU. Parameters from multitest_multi_agent_replica.sh
# ============================================================
echo ""
echo "=========================================="
echo " MAC-Ego3D Baseline"
echo "=========================================="

MACEGO_DIR="${ROOT}/repos/MAC-Ego3D"
MACEGO_OUT="${OUTPUT_BASE}/mac_ego3d"

if [ -f "${MACEGO_DIR}/mac_ego.py" ]; then
    for i in "${!SCENES[@]}"; do
        scene="${SCENES[$i]}"
        part1="${SCENE_PARTS_1[$i]}"
        part2="${SCENE_PARTS_2[$i]}"
        out_dir="${MACEGO_OUT}/${scene}"
        mkdir -p "${out_dir}"

        datasets="${DATA_DIR}/${scene}/${part1};${DATA_DIR}/${scene}/${part2}"

        echo "--- MAC-Ego3D: ${scene} ---"
        (
            cd "${MACEGO_DIR}"
            PYTHONPATH="" python -W ignore mac_ego.py \
                --dataset_path "${datasets}" \
                --config configs/Replica/caminfo.txt \
                --output_path "${out_dir}" \
                --keyframe_th 0.7 \
                --knn_maxd 99999.0 \
                --overlapped_th 5e-4 \
                --max_correspondence_distance 0.02 \
                --trackable_opacity_th 0.05 \
                --overlapped_th2 5e-5 \
                --downsample_rate 10 \
                --cuda 0 \
                --mu 1 \
                --noise 0 \
                --lc_freq 150 \
                --post_opt 1 \
                --save_results \
                2>&1 | tee "${out_dir}/run.log"
        ) || echo "WARNING: MAC-Ego3D failed on ${scene} (exit $?). Continuing..."
    done
else
    echo "SKIP: MAC-Ego3D not found at ${MACEGO_DIR}"
fi

echo ""
echo "=========================================="
echo " Baselines complete. Outputs in: ${OUTPUT_BASE}"
echo "=========================================="
