#!/bin/bash
# ================================================================
# DUAG-SLAM Project Setup Script
# ================================================================
# Run this ONCE on a fresh machine to set up everything.
# Prerequisites: git, conda, CUDA 11.8+, cmake, build-essential
# ================================================================

set -e  # Stop on any error

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "============================================"
echo "DUAG-SLAM Setup — Project Root: $PROJECT_ROOT"
echo "============================================"

# ──────────────────────────────────────────────
# STEP 1: Clone all repos into /repos (READ-ONLY reference)
# ──────────────────────────────────────────────
echo ""
echo "[Step 1/6] Cloning reference repositories into repos/..."
mkdir -p "$PROJECT_ROOT/repos"
cd "$PROJECT_ROOT/repos"

# MonoGS — our Layer 1 backbone (Gaussian Splatting SLAM)
if [ ! -d "MonoGS" ]; then
    echo "  Cloning MonoGS..."
    git clone --recursive https://github.com/muskie82/MonoGS.git
else
    echo "  MonoGS already exists, skipping."
fi

# MAC-Ego3D — we extract intra-agent consensus + association from here
if [ ! -d "MAC-Ego3D" ]; then
    echo "  Cloning MAC-Ego3D..."
    git clone --recursive https://github.com/Xiaohao-Xu/MAC-Ego3D.git
else
    echo "  MAC-Ego3D already exists, skipping."
fi

# DPGO — distributed pose graph optimization (C++ library)
if [ ! -d "dpgo" ]; then
    echo "  Cloning DPGO..."
    git clone https://github.com/mit-acl/dpgo.git
else
    echo "  DPGO already exists, skipping."
fi

# Swarm-SLAM — P2P communication infrastructure
if [ ! -d "Swarm-SLAM" ]; then
    echo "  Cloning Swarm-SLAM..."
    git clone https://github.com/MISTLab/Swarm-SLAM.git
else
    echo "  Swarm-SLAM already exists, skipping."
fi

# MAGiC-SLAM — centralized baseline for comparison
if [ ! -d "MAGiC-SLAM" ]; then
    echo "  Cloning MAGiC-SLAM..."
    git clone --recursive https://github.com/VladimirYugay/MAGiC-SLAM.git
else
    echo "  MAGiC-SLAM already exists, skipping."
fi

echo "  All repos cloned."

# ──────────────────────────────────────────────
# STEP 2: Create conda environment
# ──────────────────────────────────────────────
echo ""
echo "[Step 2/6] Creating conda environment 'duag'..."
cd "$PROJECT_ROOT"

if conda info --envs | grep -q "duag"; then
    echo "  Environment 'duag' already exists, skipping."
else
    conda create -n duag python=3.10 -y
fi

echo "  Activate with: conda activate duag"
echo "  Then run: pip install -r requirements.txt"

# ──────────────────────────────────────────────
# STEP 3: Create the core project structure
# ──────────────────────────────────────────────
echo ""
echo "[Step 3/6] Creating project directory structure..."

# /extracted — components we surgically copy from repos
mkdir -p "$PROJECT_ROOT/extracted/gs_slam/rasterizer"
mkdir -p "$PROJECT_ROOT/extracted/mac_consensus"
mkdir -p "$PROJECT_ROOT/extracted/dpgo_wrapper/build"
mkdir -p "$PROJECT_ROOT/extracted/swarm_comm"

# /core — 100% our novel code
mkdir -p "$PROJECT_ROOT/core/uncertainty"
mkdir -p "$PROJECT_ROOT/core/consensus"
mkdir -p "$PROJECT_ROOT/core/pipeline"
mkdir -p "$PROJECT_ROOT/core/utils"

# /experiments — per-dataset experiment configs
mkdir -p "$PROJECT_ROOT/experiments/replica_multiagent"
mkdir -p "$PROJECT_ROOT/experiments/aria_multiagent"
mkdir -p "$PROJECT_ROOT/experiments/s3e"
mkdir -p "$PROJECT_ROOT/experiments/kimera_multi"
mkdir -p "$PROJECT_ROOT/experiments/subtmrs"
mkdir -p "$PROJECT_ROOT/experiments/euroc_heterogeneous"

# /baselines — untouched copies for fair comparison
mkdir -p "$PROJECT_ROOT/baselines"

# /tests
mkdir -p "$PROJECT_ROOT/tests"

# /data — where datasets get downloaded
mkdir -p "$PROJECT_ROOT/data"

# /outputs — experiment results
mkdir -p "$PROJECT_ROOT/outputs"

echo "  Directory structure created."

# ──────────────────────────────────────────────
# STEP 4: Create __init__.py files
# ──────────────────────────────────────────────
echo ""
echo "[Step 4/6] Creating Python package files..."

touch "$PROJECT_ROOT/extracted/__init__.py"
touch "$PROJECT_ROOT/extracted/gs_slam/__init__.py"
touch "$PROJECT_ROOT/extracted/mac_consensus/__init__.py"
touch "$PROJECT_ROOT/extracted/dpgo_wrapper/__init__.py"
touch "$PROJECT_ROOT/extracted/swarm_comm/__init__.py"

touch "$PROJECT_ROOT/core/__init__.py"
touch "$PROJECT_ROOT/core/uncertainty/__init__.py"
touch "$PROJECT_ROOT/core/consensus/__init__.py"
touch "$PROJECT_ROOT/core/pipeline/__init__.py"
touch "$PROJECT_ROOT/core/utils/__init__.py"

echo "  Package files created."

# ──────────────────────────────────────────────
# STEP 5: Create symlinks for baselines
# ──────────────────────────────────────────────
echo ""
echo "[Step 5/6] Linking baselines..."

ln -sf "$PROJECT_ROOT/repos/MAGiC-SLAM" "$PROJECT_ROOT/baselines/MAGiC-SLAM"
ln -sf "$PROJECT_ROOT/repos/MAC-Ego3D"  "$PROJECT_ROOT/baselines/MAC-Ego3D"

echo "  Baselines linked (symlinks, not copies)."

# ──────────────────────────────────────────────
# STEP 6: Print extraction instructions
# ──────────────────────────────────────────────
echo ""
echo "[Step 6/6] Setup complete!"
echo ""
echo "============================================"
echo "NEXT STEPS — Run these in order:"
echo "============================================"
echo ""
echo "1. conda activate duag"
echo "2. pip install -r requirements.txt"
echo "3. python scripts/extract_monogs.py      ← Extracts Layer 1 files from MonoGS"
echo "4. python scripts/extract_macego.py      ← Extracts consensus from MAC-Ego3D"
echo "5. python scripts/build_dpgo_wrapper.py  ← Compiles DPGO + pybind11 bridge"
echo "6. python scripts/extract_swarm.py       ← Extracts P2P comm from Swarm-SLAM"
echo "7. python tests/test_all_extractions.py  ← Verifies everything works"
echo ""
echo "Then start writing YOUR code in /core !"
echo "============================================"
