import torch
import cv2
import numpy as np
import os
import sys
from torchvision import transforms

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
from test_video_model import DeepfakeDetector, get_transforms, DEVICE, FRAMES_PER_VIDEO

def test_on_video(video_path, model):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    
    transform = get_transforms()
    tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out, _ = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        fake_prob = probs[1].item()
        real_prob = probs[0].item()
        
    print(f"Video: {os.path.basename(video_path)}")
    print(f"  Raw Output: {out.cpu().numpy()}")
    print(f"  Probs: Real={real_prob:.4f}, Fake={fake_prob:.4f}")
    print(f"  Verdict: {'FAKE' if fake_prob > 0.5 else 'REAL'}")

# Load model
model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
model_path = 'backend/video_model_best.pth'
if not os.path.exists(model_path): model_path = 'backend/video_model_final.pth'

if os.path.exists(model_path):
    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Test on some samples from the dataset
    dataset_base = 'dataset/SDFVD/SDFVD'
    fake_dir = os.path.join(dataset_base, 'videos_fake')
    real_dir = os.path.join(dataset_base, 'videos_real')
    
    print("\n--- TESTING FAKE SAMPLES ---")
    if os.path.exists(fake_dir):
        fakes = [f for f in os.listdir(fake_dir) if f.endswith('.mp4')]
        for f in fakes[:2]:
            test_on_video(os.path.join(fake_dir, f), model)
            
    print("\n--- TESTING REAL SAMPLES ---")
    if os.path.exists(real_dir):
        reals = [f for f in os.listdir(real_dir) if f.endswith('.mp4')]
        for r in reals[:2]:
            test_on_video(os.path.join(real_dir, r), model)
else:
    print("Model weights not found!")
