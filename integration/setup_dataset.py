"""
Setup script: Downloads deepfake/real image dataset from Kaggle and
reorganises it into the folder structure expected by train_image.py:

  dataset/
    images/
      real/   <- all genuine face images
      fake/   <- all deepfake images
"""
import os
import shutil

# ── 1. Download ────────────────────────────────────────────────────────────────
print("Downloading dataset from Kaggle...")
import kagglehub
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
print(f"Downloaded to: {path}")
print(f"Top-level contents: {os.listdir(path)}")

# ── 2. Inspect structure ───────────────────────────────────────────────────────
# Walk the tree and show all subdirs so we know exactly where the images live
all_subdirs = []
for root, dirs, files in os.walk(path):
    if files:
        img_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
        if img_count:
            all_subdirs.append((root, img_count))

print("\nImage-containing directories:")
for d, n in all_subdirs:
    rel = os.path.relpath(d, path)
    print(f"  {rel}  ({n} images)")

# ── 3. Create target folders ───────────────────────────────────────────────────
DEST_REAL = os.path.join("dataset", "images", "real")
DEST_FAKE = os.path.join("dataset", "images", "fake")
os.makedirs(DEST_REAL, exist_ok=True)
os.makedirs(DEST_FAKE, exist_ok=True)

# ── 4. Map source → destination ────────────────────────────────────────────────
# The manjilkarki dataset uses folder names like:
#   Dataset/Train/Real, Dataset/Train/Fake (and Test/Validation equivalents)
# We copy ALL splits into a single pool so train_image.py can re-split 80/20.

KEYWORDS_REAL = ["real", "authentic", "genuine"]
KEYWORDS_FAKE = ["fake", "deepfake", "manipulated", "synthetic"]

copied_real = 0
copied_fake = 0

for root, dirs, files in os.walk(path):
    folder_lower = os.path.basename(root).lower()
    is_real = any(k in folder_lower for k in KEYWORDS_REAL)
    is_fake = any(k in folder_lower for k in KEYWORDS_FAKE)

    if not (is_real or is_fake):
        continue  # skip non-leaf dirs

    dest = DEST_REAL if is_real else DEST_FAKE

    for fname in files:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
        src = os.path.join(root, fname)
        # Make filename unique to avoid collisions across splits
        unique_name = f"{os.path.basename(root)}_{fname}"
        dst = os.path.join(dest, unique_name)
        shutil.copy2(src, dst)
        if is_real:
            copied_real += 1
        else:
            copied_fake += 1

# ── 5. Summary ─────────────────────────────────────────────────────────────────
print(f"\nDataset setup complete!")
print(f"  Real images copied : {copied_real}  →  {DEST_REAL}")
print(f"  Fake images copied : {copied_fake}  →  {DEST_FAKE}")
print()
if copied_real == 0 and copied_fake == 0:
    print("WARNING: No images were copied. The folder names inside the zip may differ.")
    print("Listing all leaf dirs again so you can check:")
    for d, n in all_subdirs:
        print(f"  {os.path.relpath(d, path)}")
else:
    print("You can now run:  python train_image.py")
