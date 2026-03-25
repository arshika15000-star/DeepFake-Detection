"""
Setup script: Downloads The Fake or Real Audio dataset from Kaggle
and organizes it into the expected folder structure for train_audio.py.
"""
import os
import shutil
import kagglehub

print("Downloading audio dataset from Kaggle...")
# Download latest version
path = kagglehub.dataset_download("mohammedabdeldayem/the-fake-or-real-dataset")
print("Path to dataset files:", path)

DEST_REAL = os.path.join("..", "dataset", "audio", "real")
DEST_FAKE = os.path.join("..", "dataset", "audio", "fake")
os.makedirs(DEST_REAL, exist_ok=True)
os.makedirs(DEST_FAKE, exist_ok=True)

copied_real = 0
copied_fake = 0

print("Scanning for audio files...")
for root, dirs, files in os.walk(path):
    root_lower = root.lower()
    for fname in files:
        if fname.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            src = os.path.join(root, fname)
            
            # Simple heuristic
            if 'real' in root_lower or 'real' in fname.lower() or 'human' in root_lower:
                dest = DEST_REAL
                copied_real += 1
            else:
                dest = DEST_FAKE
                copied_fake += 1
                
            unique_name = f"{os.path.basename(root)}_{fname}"
            dst = os.path.join(dest, unique_name)
            shutil.copy2(src, dst)

print("\nDataset setup complete!")
print(f"  Real audio copied : {copied_real}  →  {DEST_REAL}")
print(f"  Fake audio copied : {copied_fake}  →  {DEST_FAKE}")
