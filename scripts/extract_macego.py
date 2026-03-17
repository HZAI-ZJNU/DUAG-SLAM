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
# MAC-Ego3D structure varies — we list possible paths
# After cloning, inspect with: find repos/MAC-Ego3D -name "*.py" | head -40

# Intra-agent Gaussian consensus (temporal coherence within one agent)
for candidate in [
    "src/consensus/intra_agent.py",
    "macego/consensus/intra_agent.py",
    "consensus/intra_agent.py",
]:
    if os.path.exists(os.path.join(SRC, candidate)):
        extract(candidate, "intra_agent.py",
                "Intra-agent temporal Gaussian consistency. Keep as-is.")
        break
else:
    LOG.append("  NOTE: intra_agent.py not found at expected paths.")
    LOG.append("        After cloning, run: find repos/MAC-Ego3D -name '*.py' | sort")
    LOG.append("        Then update this script with correct path.")

# Gaussian association (matching Gaussians between views/agents)
for candidate in [
    "src/consensus/association.py",
    "macego/association.py",
    "src/utils/gaussian_matching.py",
]:
    if os.path.exists(os.path.join(SRC, candidate)):
        extract(candidate, "association.py",
                "Gaussian spatial matching. We add uncertainty as input.")
        break
else:
    LOG.append("  NOTE: association.py not found at expected paths.")
    LOG.append("        Will need manual inspection of MAC-Ego3D repo structure.")

# SALAD place recognition descriptor
for candidate in [
    "src/place_recognition/salad.py",
    "macego/salad_descriptor.py",
    "submodules/salad/",
]:
    if os.path.exists(os.path.join(SRC, candidate)):
        LOG.append(f"  NOTED: SALAD descriptor found at {candidate}")
        LOG.append(f"         Will extract for TTA-PR (Contribution 3) later.")
        break

# ─── What we DO NOT take ───
LOG.append("")
LOG.append("  NOT TAKEN: inter_agent consensus / coordinator / server")
LOG.append("             Reason: THIS IS WHAT WE REPLACE with DUAG-C (our Contribution 1)")
LOG.append("  NOT TAKEN: training scripts, evaluation scripts")
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
