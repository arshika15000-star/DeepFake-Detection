import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from pathlib import Path
import time
from sklearn.model_selection import train_test_split

# Configuration
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
IMAGE_SIZE = (224, 224)
FRAMES_PER_VIDEO = 10  # Extract 10 frames from each video
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_ROOT = r"c:\Users\Vanshina Saxena\OneDrive\Desktop\Deepfake Detection\dataset\SDFVD\SDFVD"

class VideoFrameDataset(Dataset):
    """Dataset that extracts frames from videos"""
    
    def __init__(self, video_paths, auth_labels, emotion_labels, transform=None, frames_per_video=10):
        self.video_paths = video_paths
        self.auth_labels = auth_labels
        self.emotion_labels = emotion_labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
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
        
        # Initialize face detector if possible
        face_detector = None
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception:
            pass

        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face Detection & Cropping
                if face_detector:
                    results = face_detector.process(frame_rgb)
                    if results.detections:
                        # Take the first face detected
                        bbox = results.detections[0].location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                        box_w, box_h = int(bbox.width * w), int(bbox.height * h)
                        
                        # Add some padding to the crop
                        padding = int(max(box_w, box_h) * 0.2)
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(w, x + box_w + padding)
                        y2 = min(h, y + box_h + padding)
                        
                        if x2 > x1 and y2 > y1:
                            frame_rgb = frame_rgb[y1:y2, x1:x2]
                
                # Resize to standard size anyway (crop might be different)
                frame_rgb = cv2.resize(frame_rgb, (224, 224))
                frames.append(frame_rgb)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        if face_detector:
            face_detector.close()
            
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.frames_per_video]
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        auth_label = self.auth_labels[idx]
        emotion_label = self.emotion_labels[idx]
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            
            if frames is None:
                # Return a dummy tensor if video couldn't be read
                frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.frames_per_video)]
            
            # Transform frames
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            
            # Stack frames along a new dimension: (frames, channels, height, width)
            frames_tensor = torch.stack(frames)
            
        except Exception as e:
            print(f"Warning: Error processing video {video_path}: {e}")
            # Return dummy data on error
            dummy_frame = torch.zeros((3, 224, 224))
            frames_tensor = torch.stack([dummy_frame for _ in range(self.frames_per_video)])
        
        return frames_tensor, auth_label, emotion_label

def get_transforms():
    """Data augmentation and normalization"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_test_transforms():
    """Transforms for validation/test (no augmentation)"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Import the canonical model architecture from backend/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from test_video_model import DeepfakeDetector

def parse_emotion_from_filename(filename):
    """Automatically parse emotion label from filename based on keywords"""
    filename = filename.lower()
    emotions = {
        "angry": 0, "anger": 0,
        "disgust": 1,
        "fear": 2, "afraid": 2,
        "happy": 3, "joy": 3, "smile": 3,
        "neutral": 4, "calm": 4,
        "sad": 5, "sorrow": 5,
        "surprise": 6, "shock": 6
    }
    
    for key, val in emotions.items():
        if key in filename:
            return val
    return 4  # Default to Neutral if no keyword found

def load_video_dataset():
    """Load video paths, authenticity labels, and emotion labels"""
    fake_dir = os.path.join(VIDEO_ROOT, 'videos_fake')
    real_dir = os.path.join(VIDEO_ROOT, 'videos_real')
    
    video_paths = []
    auth_labels = []
    emotion_labels = []
    
    # Load fake videos (label = 1)
    if os.path.exists(fake_dir):
        for video_file in os.listdir(fake_dir):
            if video_file.endswith('.mp4'):
                v_path = os.path.join(fake_dir, video_file)
                video_paths.append(v_path)
                auth_labels.append(1)  # Fake
                emotion_labels.append(parse_emotion_from_filename(video_file))
    
    # Load real videos (label = 0)
    if os.path.exists(real_dir):
        for video_file in os.listdir(real_dir):
            if video_file.endswith('.mp4'):
                v_path = os.path.join(real_dir, video_file)
                video_paths.append(v_path)
                auth_labels.append(0)  # Real
                emotion_labels.append(parse_emotion_from_filename(video_file))
    
    print(f"Total videos loaded: {len(video_paths)}")
    print(f"Fake: {sum(auth_labels)}, Real: {len(auth_labels) - sum(auth_labels)}")
    
    # Show emotion distribution
    from collections import Counter
    emo_counts = Counter(emotion_labels)
    emo_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    print("Emotion distribution:")
    for i, name in enumerate(emo_names):
        print(f"  {name}: {emo_counts.get(i, 0)}")
    
    return video_paths, auth_labels, emotion_labels

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with Multi-Task Loss"""
    model.train()
    running_loss = 0.0
    correct_auth = 0
    correct_emo = 0
    total = 0
    
    # Emotion labels mapping (if provided by dataset)
    # If dataset doesn't provide them, we'll skip emotion loss backprop
    
    for batch_idx, (frames, auth_y, emo_y) in enumerate(dataloader):
        frames = frames.to(device)
        auth_y = auth_y.to(device)
        emo_y = emo_y.to(device)
        
        optimizer.zero_grad()
        
        use_cuda = device.type == 'cuda'
        
        if use_cuda and scaler:
            with torch.amp.autocast('cuda'):
                auth_out, emo_out = model(frames)
                loss_auth = criterion(auth_out, auth_y)
                loss_emo = criterion(emo_out, emo_y)
                loss = loss_auth + 0.5 * loss_emo # Combine losses
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            auth_out, emo_out = model(frames)
            loss_auth = criterion(auth_out, auth_y)
            loss_emo = criterion(emo_out, emo_y)
            loss = loss_auth + 0.5 * loss_emo
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, pred_auth = auth_out.max(1)
        _, pred_emo = emo_out.max(1)
        total += auth_y.size(0)
        correct_auth += pred_auth.eq(auth_y).sum().item()
        correct_emo += pred_emo.eq(emo_y).sum().item()
        
        if (batch_idx + 1) % 5 == 0:
            avg_loss = running_loss / 5
            acc_auth = 100. * correct_auth / total
            acc_emo = 100. * correct_emo / total
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f} | Auth: {acc_auth:.1f}% | Emo: {acc_emo:.1f}%", flush=True)
            running_loss = 0.0
    
    epoch_acc = 100. * correct_auth / total
    return epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate model on Multi-Task Performance"""
    model.eval()
    running_loss = 0.0
    correct_auth = 0
    correct_emo = 0
    total = 0
    
    with torch.no_grad():
        for frames, auth_y, emo_y in dataloader:
            frames = frames.to(device)
            auth_y = auth_y.to(device)
            emo_y = emo_y.to(device)
            
            auth_out, emo_out = model(frames)
            loss_auth = criterion(auth_out, auth_y)
            loss_emo = criterion(emo_out, emo_y)
            loss = loss_auth + 0.5 * loss_emo
            
            running_loss += loss.item()
            _, pred_auth = auth_out.max(1)
            _, pred_emo = emo_out.max(1)
            total += auth_y.size(0)
            correct_auth += pred_auth.eq(auth_y).sum().item()
            correct_emo += pred_emo.eq(emo_y).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    acc_auth = 100. * correct_auth / total
    acc_emo = 100. * correct_emo / total
    
    return avg_loss, acc_auth, acc_emo

def main():
    print(f"Using device: {DEVICE}")
    print(f"Training configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Frames per video: {FRAMES_PER_VIDEO}")
    print()
    
    # Load dataset
    print("Loading video dataset with Multi-Task analysis...")
    video_paths, auth_labels, emotion_labels = load_video_dataset()
    
    if len(video_paths) == 0:
        print("Error: No videos found!")
        return
    
    # Split into train and validation
    train_paths, val_paths, train_auth, val_auth, train_emo, val_emo = train_test_split(
        video_paths, auth_labels, emotion_labels, test_size=0.2, random_state=42, stratify=auth_labels
    )
    
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    print()
    
    # Create datasets
    train_dataset = VideoFrameDataset(train_paths, train_auth, train_emo,
                                     transform=get_transforms(), 
                                     frames_per_video=FRAMES_PER_VIDEO)
    val_dataset = VideoFrameDataset(val_paths, val_auth, val_emo,
                                   transform=get_test_transforms(), 
                                   frames_per_video=FRAMES_PER_VIDEO)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    
    # Create model
    print("Building model...")
    model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=2)
    
    # Mixed precision training
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    
    print("Starting training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        
        # Validate
        val_loss, val_acc_auth, val_acc_emo = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Authentication Acc: {val_acc_auth:.2f}%")
        print(f"  Val Emotion Acc: {val_acc_emo:.2f}%")
        
        # Learning rate scheduling based on authenticity performance
        scheduler.step(val_acc_auth)
        
        # Save checkpoint
        checkpoint_path = f"video_model_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc_auth': val_acc_auth,
            'val_acc_emo': val_acc_emo,
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_acc_auth > best_val_acc:
            best_val_acc = val_acc_auth
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc_auth,
            }, "video_model_best.pth")
            print(f"  ⭐ New best model! Auth Accuracy: {val_acc_auth:.2f}%")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), "video_model_final.pth")
    print("Final model saved to video_model_final.pth")

if __name__ == "__main__":
    main()
