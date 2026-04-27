import torch
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\backend")
from test_video_model import DeepfakeDetector, get_transforms, extract_frames

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\video_model_best.pth"

def run_sanity_check():
    print("Running Model Sanity Check...")
    
    model = DeepfakeDetector(num_frames=10).to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE)['model_state_dict'], strict=False)
    model.eval()
    
    # Check current test videos
    base = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD"
    test_videos = []
    for sub in ['videos_fake', 'videos_real']:
        folder = os.path.join(base, sub)
        if os.path.exists(folder):
            test_videos.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.mov'))][:5])
    
    if not test_videos:
        print("No test videos found.")
        return

    all_probs = []
    transform = get_transforms()

    with torch.no_grad():
        for vid_path in test_videos:
            frames = extract_frames(vid_path, 10)
            if frames:
                tensors = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(DEVICE)
                output, _ = model(tensors)
                prob = torch.softmax(output, dim=1)[0][1].item()
                all_probs.append(prob)
                print(f" Video: {os.path.basename(vid_path)} | Conf: {prob:.4f}")

    if all_probs:
        print("\nSUMMARY statistics:")
        print(f" Min prob: {min(all_probs):.3f}")
        print(f" Max prob: {max(all_probs):.3f}")
        print(f" Mean prob: {np.mean(all_probs):.3f}")
        
        if 0.45 <= np.mean(all_probs) <= 0.65:
            print("\nWARNING: Model is showing high uncertainty (collapsed).")
            print("Action: Complete the Celeb-DF-v2 download and re-train with Frozen Backbone.")
        else:
            print("\nModel shows some discriminative signal.")

if __name__ == "__main__":
    run_sanity_check()
