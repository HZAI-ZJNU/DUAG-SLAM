# DUAG-SLAM: Decentralized Uncertainty-Aware Gaussian SLAM

## Quick Start (5 minutes)

```bash
# 1. Clone this project
git clone <your-repo-url> DUAG-SLAM
cd DUAG-SLAM

# 2. Run setup (clones all 5 reference repos into /repos)
chmod +x setup.sh
./setup.sh

# 3. Create conda environment
conda activate duag
pip install -r requirements.txt

# 4. Run tests on core code (no repos needed yet)
python -m pytest tests/test_consensus.py -v

# 5. Extract components from repos
python scripts/extract_monogs.py
python scripts/extract_macego.py
```

## How It Works

```
DUAG-SLAM/
├── repos/          ← 5 cloned repos (READ-ONLY, never edit these)
│   ├── MonoGS/
│   ├── MAC-Ego3D/
│   ├── dpgo/
│   ├── Swarm-SLAM/
│   └── MAGiC-SLAM/
│
├── extracted/      ← Surgically copied files from repos (thin adapters)
│   ├── gs_slam/         ← From MonoGS
│   ├── mac_consensus/   ← From MAC-Ego3D
│   ├── dpgo_wrapper/    ← From DPGO (C++ with pybind11)
│   └── swarm_comm/      ← From Swarm-SLAM
│
├── core/           ← 100% OUR CODE (all novel contributions)
│   ├── types.py              ← Data structures everything shares
│   ├── uncertainty/          ← Sub-contribution 1.1 (Hessian uncertainty)
│   ├── consensus/            ← Sub-contributions 1.2-1.3 (DUAG-C optimizer)
│   └── pipeline/             ← Layer orchestration
│
├── experiments/    ← Per-dataset experiment scripts
├── baselines/      ← Symlinks to untouched repos for comparison
└── tests/          ← Unit tests (run BEFORE integration)
```

## The Rule

- `/repos` = grocery store. You read, you study, you NEVER edit.
- `/extracted` = your fridge. Only what you need, prepped with adapters.
- `/core` = your cooking. Everything novel lives here. Copilot helps here.
- `/baselines` = restaurant next door. Run their code untouched for comparison.

## Working with GitHub Copilot

Open the DUAG-SLAM folder in VS Code. Copilot can see:
- `/repos/MonoGS/` to understand how the rasterizer works
- `/repos/MAC-Ego3D/` to understand how Gaussian consensus works
- `/core/` where you write your novel code

When writing in `/core`, Copilot will suggest code that references
patterns from the repos. This is the intended workflow.
