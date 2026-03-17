#!/usr/bin/env python3
"""
Extract MonoGS components into extracted/gs_slam/
This copies ONLY what we need and logs every file copied.
"""
import shutil, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "repos", "MonoGS")
DST = os.path.join(ROOT, "extracted", "gs_slam")
LOG = []

def extract(src_relative, dst_relative, note):
    """Copy one file and log it."""
    src_path = os.path.join(SRC, src_relative)
    dst_path = os.path.join(DST, dst_relative)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        LOG.append(f"  COPIED: {src_relative} → extracted/gs_slam/{dst_relative}")
        LOG.append(f"          Reason: {note}")
    else:
        LOG.append(f"  WARNING: {src_path} not found!")

print("=" * 60)
print("Extracting from MonoGS → extracted/gs_slam/")
print("=" * 60)

# ─── What we take ───

# 1. The Gaussian model (how Gaussians are stored and managed)
extract("gaussian_splatting/scene/gaussian_model.py",
        "gaussian_model.py",
        "Core Gaussian primitive storage. We extend this with uncertainty field.")

# 2. The SLAM frontend (tracking: estimate camera pose from Gaussians)
extract("slam/slam_frontend.py",
        "tracker.py",
        "Camera pose estimation via differentiable rendering. Used as-is.")

# 3. The SLAM backend (mapping: create/optimize/prune Gaussians)
extract("slam/slam_backend.py",
        "mapper.py",
        "Gaussian creation, optimization, pruning. We hook uncertainty update here.")

# 4. The differentiable rasterizer (CUDA extension — copy entire submodule)
rasterizer_src = os.path.join(SRC, "submodules", "diff-gaussian-rasterization")
rasterizer_dst = os.path.join(DST, "rasterizer")
if os.path.exists(rasterizer_src):
    # Copy the whole directory
    if os.path.exists(rasterizer_dst):
        shutil.rmtree(rasterizer_dst)
    shutil.copytree(rasterizer_src, rasterizer_dst)
    LOG.append(f"  COPIED: submodules/diff-gaussian-rasterization/ → extracted/gs_slam/rasterizer/")
    LOG.append(f"          Reason: CUDA rasterizer with analytical Jacobians. Compiled separately.")
else:
    LOG.append(f"  WARNING: Rasterizer submodule not found (need --recursive clone)")

# 5. Simple-knn (CUDA extension for nearest neighbor)
simpleknn_src = os.path.join(SRC, "submodules", "simple-knn")
simpleknn_dst = os.path.join(DST, "simple_knn")
if os.path.exists(simpleknn_src):
    if os.path.exists(simpleknn_dst):
        shutil.rmtree(simpleknn_dst)
    shutil.copytree(simpleknn_src, simpleknn_dst)
    LOG.append(f"  COPIED: submodules/simple-knn/ → extracted/gs_slam/simple_knn/")
    LOG.append(f"          Reason: KNN for Gaussian initialization from point clouds.")

# 6. Utility functions
extract("utils/slam_utils.py",
        "slam_utils.py",
        "Helper functions for SLAM (image processing, pose conversion).")

# ─── What we DO NOT take (and why) ───
LOG.append("")
LOG.append("  NOT TAKEN: slam/slam_loop_closure.py")
LOG.append("             Reason: We replace with TTA-PR (Contribution 3)")
LOG.append("  NOT TAKEN: viewer/ (entire directory)")
LOG.append("             Reason: We build our own visualization")
LOG.append("  NOT TAKEN: configs/ scripts/ eval/")
LOG.append("             Reason: We write our own pipeline orchestration")

# ─── Print log ───
print()
for line in LOG:
    print(line)

# ─── Save extraction log ───
log_path = os.path.join(ROOT, "docs", "EXTRACTION_LOG.md")
with open(log_path, "a") as f:
    f.write("\n\n## MonoGS Extraction\n")
    f.write(f"Source: repos/MonoGS/\n")
    f.write(f"Destination: extracted/gs_slam/\n\n")
    for line in LOG:
        f.write(line + "\n")

print()
print(f"Extraction log appended to docs/EXTRACTION_LOG.md")
print("DONE. Next: python scripts/extract_macego.py")
