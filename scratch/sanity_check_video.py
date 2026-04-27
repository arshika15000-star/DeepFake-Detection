import torch
import sys
import os

# Add backend to path to import model
sys.path.append(r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\backend")
from test_video_model import DeepfakeDetector, get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\video_model_best.pth"

def test_single_video(video_path):
    from test_video_model import extract_frames
    model = DeepfakeDetector(num_frames=10).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)['model_state_dict'], strict=False)
    model.eval()
    
    transform = get_transforms()
    frames = extract_frames(video_path, 10)
    if frames is None:
        return "Failed to extract frames"
    
    frames_tensor = [transform(f) for f in frames]
    frames_tensor = torch.stack(frames_tensor).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        auth_out, _ = model(frames_tensor)
        probs = torch.softmax(auth_out, dim=1)[0]
    
    return {
        "logits": auth_out.cpu().numpy().tolist(),
        "probs": probs.cpu().numpy().tolist(),
        "pred_class": int(auth_out.argmax(dim=1).item())
    }

fake_vid = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD\videos_fake\deepfake_1.mp4"
real_vid = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD\videos_real\video_1.mp4"

print("Testing FAKE video...")
print(test_single_video(fake_vid))
print("\nTesting REAL video...")
print(test_single_video(real_vid))
