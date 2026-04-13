import torch
import torchaudio
import os
import sys
import numpy as np
import librosa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'audio')
sys.path.insert(0, BACKEND_DIR)

from text_audio_models import AudioDiscriminator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 16000

def load_audio_model(model_name='audio_model_best.pth'):
    model = AudioDiscriminator(feature_dim=768).to(DEVICE)
    model_path = os.path.join(BACKEND_DIR, model_name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model {model_name} not found.")
        return None
    model.eval()
    return model

def run_test():
    model = load_audio_model()
    if not model: return

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec = bundle.get_model().to(DEVICE).eval()

    test_samples = []
    print("\nScanning for test samples...")
    for label_str, label_int in [('real', 0), ('fake', 1)]:
        dir_path = os.path.join(DATASET_DIR, label_str)
        if not os.path.exists(dir_path): continue
        files = [f for f in os.listdir(dir_path) if f.endswith(('.wav', '.mp3'))][200:300] # Use a different subset for testing
        for f in files:
            test_samples.append((os.path.join(dir_path, f), label_int))

    if not test_samples:
        print("No test samples found.")
        return

    print(f"Testing on {len(test_samples)} samples...")
    
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for i, (path, label) in enumerate(test_samples):
            if (i+1) % 50 == 0: print(f"  Tested {i+1}/{len(test_samples)}...")
            try:
                audio, sr = librosa.load(path, sr=SAMPLE_RATE)
                wf = torch.from_numpy(audio).unsqueeze(0).float().to(DEVICE)
                feats, _ = wav2vec(wf)
                pooled = feats.squeeze(0).mean(dim=0).unsqueeze(0)
                logits = model(pooled)
                probs = torch.softmax(logits, dim=1)[0]
                
                y_true.append(label)
                y_pred.append(logits.argmax(dim=1).item())
                y_scores.append(probs[1].item()) # Fake probability
            except Exception as e:
                pass

    print("\n" + "="*30)
    print("AUDIO DETECTION FINAL AUDIT")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    auc = roc_auc_score(y_true, y_scores)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    run_test()
