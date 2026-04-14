import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import os
import numpy as np
from pathlib import Path

# Configuration
IMAGE_SIZE = (224, 224)
FRAMES_PER_VIDEO = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_ROOT = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD"

class DeepfakeDetector(nn.Module):
    """Refined Model with ResNet50 + Bi-LSTM for Deepfake Detection"""
    
    def __init__(self, num_frames=10):
        super(DeepfakeDetector, self).__init__()
        
        # Backbone: ResNet50
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal: High-Capacity Bidirectional LSTM
        # ResNet50 outputs 2048 features
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        
        self.dropout = nn.Dropout(p=0.5)
        
        # Authenticity Classification head (Sequential with BatchNorm for stability)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  # 512*2 because bidirectional = 1024
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)         # 2 classes: REAL / FAKE
        )
        
        # Emotion Classification head (Optional/Secondary)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )
        
        # Legacy pointers to keep compatibility if needed elsewhere
        self.fc1 = None 
        self.fc2 = None

    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.size()
        
        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features from each frame
        features = self.feature_extractor(x)  # (batch*frames, 2048, 7, 7)
        features = self.avgpool(features)     # (batch*frames, 2048, 1, 1)
        features = features.view(batch_size, num_frames, 2048)
        
        # Process temporal sequence with LSTM
        lstm_out, _ = self.lstm(features)
        
        # Use only the last hidden state for classification (contains both directions)
        last_hidden = lstm_out[:, -1, :] # (batch, 1024)
        last_hidden = self.dropout(last_hidden)
        
        authen_out = self.classifier(last_hidden)
        emotion_out = self.emotion_classifier(last_hidden)
        
        return authen_out, emotion_out

def get_transforms():
    """Transforms for test (no augmentation)"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_frames(video_path, frames_per_video=10):
    """Extract evenly spaced frames from a video, cropping to the face if detected"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
    # Initialize face detector
    face_detector = None
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    except Exception:
        pass

    # Calculate frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face Detection
            if face_detector:
                results = face_detector.process(frame_rgb)
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    box_w, box_h = int(bbox.width * w), int(bbox.height * h)
                    
                    padding = int(max(box_w, box_h) * 0.2)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(w, x + box_w + padding)
                    y2 = min(h, y + box_h + padding)
                    
                    if x2 > x1 and y2 > y1:
                        frame_rgb = frame_rgb[y1:y2, x1:x2]
            
            # Final Resize
            frame_rgb = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_rgb)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    cap.release()
    if face_detector:
        face_detector.close()
    
    while len(frames) < frames_per_video:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    return frames[:frames_per_video]

def predict_video(model, video_path, transform):
    """Predict if a video is fake or real"""
    # Extract frames
    frames = extract_frames(video_path, FRAMES_PER_VIDEO)
    
    if frames is None:
        return None, None
    
    # Transform frames
    frames_tensor = [transform(frame) for frame in frames]
    frames_tensor = torch.stack(frames_tensor).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    frames_tensor = frames_tensor.to(DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        authen_out, emotion_out = model(frames_tensor)
        
        # Authenticity
        probabilities = torch.softmax(authen_out, dim=1)
        predicted_class = authen_out.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        # Emotion
        emotion_probs = torch.softmax(emotion_out, dim=1)
        emotion_idx = emotion_out.argmax(dim=1).item()
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        predicted_emotion = emotion_labels[emotion_idx]
        emotion_confidence = emotion_probs[0][emotion_idx].item()
    
    return predicted_class, confidence, predicted_emotion, emotion_confidence

def test_model(model_path="video_model_best.pth"):
    """Test the trained model on the validation set"""
    print(f"Using device: {DEVICE}")
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Model file {model_path} not found!")
        return
    
    model.eval()
    transform = get_transforms()
    
    # Test on some videos
    fake_dir = os.path.join(VIDEO_ROOT, 'videos_fake')
    real_dir = os.path.join(VIDEO_ROOT, 'videos_real')
    
    print("\n" + "=" * 70)
    print("Testing on sample videos...")
    print("=" * 70)
    
    # Test fake videos
    if os.path.exists(fake_dir):
        fake_videos = [f for f in os.listdir(fake_dir) if f.endswith('.mp4')][:5]
        print("\n[VIDEO] Testing FAKE videos:")
        print("-" * 70)
        
        fake_correct = 0
        for video_file in fake_videos:
            video_path = os.path.join(fake_dir, video_file)
            predicted_class, confidence, predicted_emotion, emotion_confidence = predict_video(model, video_path, transform)
            
            if predicted_class is not None:
                label = "FAKE" if predicted_class == 1 else "REAL"
                is_correct = predicted_class == 1
                fake_correct += is_correct
                
                status = "[OK]" if is_correct else "[FAIL]"
                print(f"{status} {video_file:20s} | Pred: {label:4s} | Emotion: {predicted_emotion:8s} | Conf: {confidence:.2%}")
        
        print(f"\nFake videos accuracy: {fake_correct}/{len(fake_videos)} ({100*fake_correct/len(fake_videos):.1f}%)")
    
    # Test real videos
    if os.path.exists(real_dir):
        real_videos = [f for f in os.listdir(real_dir) if f.endswith('.mp4')][:5]
        print("\n[VIDEO] Testing REAL videos:")
        print("-" * 70)
        
        real_correct = 0
        for video_file in real_videos:
            video_path = os.path.join(real_dir, video_file)
            predicted_class, confidence, predicted_emotion, emotion_confidence = predict_video(model, video_path, transform)
            
            if predicted_class is not None:
                label = "FAKE" if predicted_class == 1 else "REAL"
                is_correct = predicted_class == 0
                real_correct += is_correct
                
                status = "[OK]" if is_correct else "[FAIL]"
                print(f"{status} {video_file:20s} | Pred: {label:4s} | Emotion: {predicted_emotion:8s} | Conf: {confidence:.2%}")
        
        print(f"\nReal videos accuracy: {real_correct}/{len(real_videos)} ({100*real_correct/len(real_videos):.1f}%)")
    
    print("\n" + "=" * 70)

def test_single_video(video_path, model_path="video_model_best.pth"):
    """Test a single video file"""
    print(f"Using device: {DEVICE}")
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Model file {model_path} not found!")
        return
    
    model.eval()
    transform = get_transforms()
    
    print(f"\nTesting video: {video_path}")
    predicted_class, confidence, predicted_emotion, emotion_confidence = predict_video(model, video_path, transform)
    
    if predicted_class is not None:
        label = "FAKE" if predicted_class == 1 else "REAL"
        print(f"\nAuthenticity: {label} (Conf: {confidence:.2%})")
        print(f"Detected Emotion: {predicted_emotion} (Conf: {emotion_confidence:.2%})")
        
        if predicted_class == 1:
            print("[WARN]  This video appears to be a DEEPFAKE")
        else:
            print("[OK] This video appears to be REAL")
    else:
        print("Error: Could not process video")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test a single video
        video_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "video_model_best.pth"
        test_single_video(video_path, model_path)
    else:
        # Test on validation set
        test_model()
