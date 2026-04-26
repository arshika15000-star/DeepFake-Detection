import os
import shutil
import tempfile

def main() -> None:
    print("TARGETING LARGE-SCALE MULTIMODAL DATASET...")
    import kagglehub
    
    print("Scanning for local datasets...")
    path = "C:/Users/Vanshina Saxena/.cache/kagglehub/datasets/reubensuju/celeb-df-v2/versions/1"
    
    if not os.path.exists(path):
        try:
           print("Local simongraves not found. Attempting kagglehub pull...")
           path = kagglehub.dataset_download("simongraves/deepfake-dataset")
        except Exception:
           path = kagglehub.dataset_download("yashasvi0/deepfake-detection-dataset")
        
    print(f"Source Data: {path}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEST_REAL = os.path.join(base_dir, "dataset", "SDFVD", "SDFVD", "videos_real")
    DEST_FAKE = os.path.join(base_dir, "dataset", "SDFVD", "SDFVD", "videos_fake")
    os.makedirs(DEST_REAL, exist_ok=True)
    os.makedirs(DEST_FAKE, exist_ok=True)

    copied_real = 0
    copied_fake = 0

    print("Aggressive scraping enabled. Scanning for video artifacts...")

    for root, dirs, files in os.walk(path):
        for fname in files:
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
                src = os.path.join(root, fname)

                # Check ALL parent folders in the path AND the filename
                path_str = f"{root}/{fname}".lower().replace("\\", "/")

                # Check the folder name and the full path string
                path_str = f"{root}/{fname}".lower().replace("\\", "/")
                parent_folder = os.path.basename(root).lower()

                fake_keys = ('fake', 'deepfake', 'manipulated', 'altered', 'forged', 'synthesis', 'synthetic', 'spoof', 'df_')
                real_keys = ('real', 'original', 'authentic', 'genuine', 'pristine', 'raw', 'source', 'celeb-real')

                # Check if the folder itself or the filename indicates the label
                is_fake = any(key in parent_folder or key in fname.lower() for key in fake_keys)
                is_real = any(key in parent_folder or key in fname.lower() for key in real_keys)

                if is_fake and not is_real:
                    dest = DEST_FAKE
                    copied_fake += 1
                elif is_real and not is_fake:
                    dest = DEST_REAL
                    copied_real += 1
                elif "video" in parent_folder:
                    dest = DEST_REAL
                    copied_real += 1
                elif "deepfake" in parent_folder:
                    dest = DEST_FAKE
                    copied_fake += 1
                else:
                    # Generic check for any other part of the path
                    is_fake_path = any(key in path_str for key in fake_keys if key != 'fake') # skip 'fake' to avoid matching root folder
                    if is_fake_path:
                         dest = DEST_FAKE
                         copied_fake += 1
                    else:
                         continue

                # Ensure unique names even if datasets use same filenames
                unique_name = f"{os.path.basename(root)}_{fname}"
                dst = os.path.join(dest, unique_name)
                
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    print("\nDATASET SYNCHRONIZED!")
    print(f"  REAL VIDEOS CAPTURED: {copied_real}")
    print(f"  FAKE VIDEOS CAPTURED: {copied_fake}")
    print("-" * 40)
    
    # Verification Check
    real_count = len(os.listdir(DEST_REAL))
    fake_count = len(os.listdir(DEST_FAKE))
    print(f"Final Inventory -> Real: {real_count} | Fake: {fake_count}")
    
    if real_count > 0 and fake_count > 0:
        ratio = min(real_count, fake_count) / max(real_count, fake_count)
        if ratio < 0.7:
            print("WARNING: Dataset is heavily imbalanced. Consider class_weight in training.")
        else:
            print("Dataset looks balanced.")
    
    if real_count < 10 or fake_count < 10:
         print("WARNING: Data pool still too low for deep learning.")
         print("TIP: Go to Kaggle.com and search for 'Deepfake Detection Challenge'.")
    else:
         print("SUFFICIENT DATA FOUND. You can now re-run: python integration/train_video.py")

if __name__ == "__main__":
    main()
