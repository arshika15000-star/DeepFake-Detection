"""
Setup script: Downloads Simon Graves deepfake video dataset from Kaggle
and organizes it into the folder structure expected by train_video.py:

  dataset/
    SDFVD/
      SDFVD/
        videos_real/   <- authentic videos
        videos_fake/   <- deepfake videos
"""
import os
import shutil

# ── 1. Download ────────────────────────────────────────────────────────────────
print("Downloading video dataset from Kaggle...")
import kagglehub
path = kagglehub.dataset_download("simongraves/deepfake-dataset")
print(f"Downloaded to: {path}")

# ── 2. Create target folders ───────────────────────────────────────────────────
DEST_REAL = os.path.join("..", "dataset", "SDFVD", "SDFVD", "videos_real")
DEST_FAKE = os.path.join("..", "dataset", "SDFVD", "SDFVD", "videos_fake")
os.makedirs(DEST_REAL, exist_ok=True)
os.makedirs(DEST_FAKE, exist_ok=True)

# ── 3. Map source → destination ────────────────────────────────────────────────
# simongraves dataset usually contains a 'video' folder inside 'deepfake' or root.
# We will search the tree for .mp4, .mov, .avi, etc.

copied_real = 0
copied_fake = 0

print("Scanning for videos...")

for root, dirs, files in os.walk(path):
    root_lower = root.lower()
    for fname in files:
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
            src = os.path.join(root, fname)
            
            # Simple heuristic since it's a small dataset:
            # If the path contains 'real', 'original', 'authentic':
            if any(k in root_lower or k in fname.lower() for k in ['real', 'original', 'authentic']):
                dest = DEST_REAL
                copied_real += 1
            else:
                dest = DEST_FAKE
                copied_fake += 1
                
            unique_name = f"{os.path.basename(root)}_{fname}"
            dst = os.path.join(dest, unique_name)
            shutil.copy2(src, dst)

# ── 4. Summary ─────────────────────────────────────────────────────────────────
print(f"\nDataset setup complete!")
print(f"  Real videos copied : {copied_real}  →  {DEST_REAL}")
print(f"  Fake videos copied : {copied_fake}  →  {DEST_FAKE}")
print()
if copied_real == 0 and copied_fake == 0:
    print("WARNING: No videos were found or copied.")
else:
    print("You can now run:  python train_video.py")
