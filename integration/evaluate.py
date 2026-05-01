import os
import sys

# Allow imports from sibling directories
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                          # integration/
sys.path.insert(0, os.path.join(_HERE, '..', 'backend'))          # backend/

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from train_video import VideoFrameDataset, get_test_transforms, VIDEO_ROOT, FRAMES_PER_VIDEO
from test_video_model import DeepfakeDetector
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_datasets(batch_size=8):
    # Gather video paths and labels similar to train_video.load_video_dataset
    fake_dir = os.path.join(VIDEO_ROOT, 'videos_fake')
    real_dir = os.path.join(VIDEO_ROOT, 'videos_real')

    paths = []
    labels = []
    if os.path.exists(fake_dir):
        for f in os.listdir(fake_dir):
            if f.endswith('.mp4'):
                paths.append(os.path.join(fake_dir, f))
                labels.append(1)
    if os.path.exists(real_dir):
        for f in os.listdir(real_dir):
            if f.endswith('.mp4'):
                paths.append(os.path.join(real_dir, f))
                labels.append(0)

    # Add dummy emotion labels for evaluation (not needed for authenticity)
    dummy_emotions = [0] * len(labels)
    dataset = VideoFrameDataset(paths, labels, dummy_emotions, transform=get_test_transforms(), frames_per_video=FRAMES_PER_VIDEO)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

def evaluate(model_path='video_model_best.pth'):
    print(f"Loading model from {model_path} on {DEVICE}")
    model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
    if not os.path.exists(model_path):
        print('Model file not found:', model_path)
        return
    ck = torch.load(model_path, map_location=DEVICE)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        model.load_state_dict(ck['model_state_dict'])
    else:
        model.load_state_dict(ck)

    model.eval()
    loader = load_datasets()

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(DEVICE)
            outputs = model(frames)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            scores = probs[:,1].cpu().numpy().tolist()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds)
            y_scores.extend(scores)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = float('nan')

    print('Evaluation results:')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1-score:  {f1:.4f}')
    print(f'  ROC AUC:   {auc:.4f}')

if __name__ == '__main__':
    evaluate()
