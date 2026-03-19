#!/usr/bin/env python3
"""
Extract MAC-Ego3D components into extracted/mac_consensus/
We take ONLY the intra-agent consensus and association logic.
We THROW AWAY the inter-agent coordinator (we replace it with DUAG-C).
"""
import shutil, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "repos", "MAC-Ego3D")
DST = os.path.join(ROOT, "extracted", "mac_consensus")
LOG = []

def extract(src_relative, dst_relative, note):
    src_path = os.path.join(SRC, src_relative)
    dst_path = os.path.join(DST, dst_relative)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        LOG.append(f"  COPIED: {src_relative} → extracted/mac_consensus/{dst_relative}")
        LOG.append(f"          Reason: {note}")
    else:
        LOG.append(f"  WARNING: {src_path} not found! Check MAC-Ego3D repo structure.")

print("=" * 60)
print("Extracting from MAC-Ego3D → extracted/mac_consensus/")
print("=" * 60)

# ─── Navigate the repo to find the right files ───
# MAC-Ego3D structure: no formal consensus module — consensus is implicit via
# overlap elimination (GICP distance thresholding) + joint rendering optimization.
# We extract the serialization utilities and the overlap elimination logic.

# Serialization / Gaussian packing (SharedGaussians, SharedTargetPoints, etc.)
extract("scene/shared_objs.py", "gaussian_utils.py",
        "Gaussian packing/unpacking (SharedGaussians, SharedTargetPoints). "
        "Used for inter-process data exchange.")

# Overlap elimination (closest to intra-agent consensus)
# eliminate_overlapped2 in mac_Tracker.py filters new points too close to existing
# map Gaussians. Also add_from_pcd2_tensor / get_trackable_gaussians_tensor from
# gaussian_model.py for the accumulation pattern.
extract("mac_Tracker.py", "intra_agent_consensus.py",
        "Contains eliminate_overlapped2() — overlap-based Gaussian deduplication. "
        "MAC-Ego3D's implicit temporal consensus mechanism.")

# Gaussian model (for add_from_pcd2_tensor, get_trackable_gaussians_tensor)
extract("scene/gaussian_model.py", "mac_gaussian_model.py",
        "GaussianModel with densification_postfix, add_from_pcd2_tensor, "
        "get_trackable_gaussians_tensor.")

# SALAD place recognition descriptor
if os.path.exists(os.path.join(SRC, "models/aggregators/salad.py")):
    LOG.append(f"  NOTED: SALAD descriptor found at models/aggregators/salad.py")
    LOG.append(f"         Will extract for TTA-PR (Contribution 3) later.")

# ─── What we DO NOT take ───
LOG.append("")
LOG.append("  NOT TAKEN: inter_agent consensus / coordinator / server")
LOG.append("             Reason: THIS IS WHAT WE REPLACE with DUAG-C (our Contribution 1)")
LOG.append("  NOT TAKEN: training scripts, evaluation scripts, VPR utils")
LOG.append("             Reason: We write our own experiment pipeline")

print()
for line in LOG:
    print(line)

log_path = os.path.join(ROOT, "docs", "EXTRACTION_LOG.md")
with open(log_path, "a") as f:
    f.write("\n\n## MAC-Ego3D Extraction\n")
    for line in LOG:
        f.write(line + "\n")

print()
print("DONE. Next: python scripts/build_dpgo_wrapper.py")
