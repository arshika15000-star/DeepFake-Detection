import os
import shutil
import tempfile

def main() -> None:
    print("🚀 TARGETING LARGE-SCALE MULTIMODAL DATASET...")
    import kagglehub
    # Using a more comprehensive dataset link if possible, or sticking to the current one with better scraping
    try:
        path = kagglehub.dataset_download("simongraves/deepfake-dataset")
    except Exception:
        print("Falling back to secondary dataset source...")
        path = kagglehub.dataset_download("yashasvi0/deepfake-detection-dataset")
        
    print(f"📦 Source Data: {path}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEST_REAL = os.path.join(base_dir, "dataset", "SDFVD", "SDFVD", "videos_real")
    DEST_FAKE = os.path.join(base_dir, "dataset", "SDFVD", "SDFVD", "videos_fake")
    os.makedirs(DEST_REAL, exist_ok=True)
    os.makedirs(DEST_FAKE, exist_ok=True)

    copied_real = 0
    copied_fake = 0

    print("🔍 Aggressive scraping enabled. Scanning for video artifacts...")

    for root, dirs, files in os.walk(path):
        for fname in files:
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
                src = os.path.join(root, fname)
                
                # AGGRESSIVE HEURISTIC:
                # Check filename AND the entire relative path for 'fake' or 'real' keywords
                rel_path = os.path.relpath(src, path).lower()
                
                is_fake = False
                if any(x in rel_path for x in ['fake', 'spoof', 'manipulated', 'deepfake', 'df', 'synth']):
                    is_fake = True
                
                # Categorize
                if is_fake:
                    dest = DEST_FAKE
                    copied_fake += 1
                else:
                    dest = DEST_REAL
                    copied_real += 1
                    
                # Ensure unique names even if datasets use same filenames (e.g. 1.mp4)
                unique_name = f"{root.replace(os.sep, '_')[-50:]}_{fname}"
                dst = os.path.join(dest, unique_name)
                
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    print(f"\n✅ DATASET SYNCHRONIZED!")
    print(f"  📂 REAL VIDEOS CAPTURED: {copied_real}")
    print(f"  📂 FAKE VIDEOS CAPTURED: {copied_fake}")
    print("-" * 40)
    
    if copied_real < 10 or copied_fake < 10:
         print("🛑 WARNING: Data pool still too low for deep learning.")
         print("TIP: Go to Kaggle.com and search for 'Deepfake Detection Challenge'.")
         print("Download the 1GB 'sample' zip and extract it into 'dataset/SDFVD/SDFVD/'.")
    else:
         print("🔥 SUFFICIENT DATA FOUND. You can now re-run: python integration/train_video.py")

if __name__ == "__main__":
    main()
