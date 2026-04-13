import torch
import torchaudio
import os
import sys
import numpy as np
import librosa
from sklearn.metrics import accuracy_score

# Set directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'audio')
sys.path.insert(0, BACKEND_DIR)

from text_audio_models import AudioDiscriminator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_audio_model():
    model = AudioDiscriminator(feature_dim=768).to(DEVICE)
    model_path = os.path.join(BACKEND_DIR, 'audio_model_best.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded existing model from {model_path}")
    else:
        print("No existing model found.")
        return None
    model.eval()
    return model

def verify():
    model = load_audio_model()
    if not model: return

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec = bundle.get_model().to(DEVICE).eval()

    print("\nVerifying on dataset/audio samples...")
    
    results = []
    
    for label_str, label_int in [('real', 0), ('fake', 1)]:
        dir_path = os.path.join(DATASET_DIR, label_str)
        files = [f for f in os.listdir(dir_path) if f.endswith(('.wav', '.mp3'))][:10]
        
        for f in files:
            path = os.path.join(dir_path, f)
            try:
                audio, sr = librosa.load(path, sr=16000)
                wf = torch.from_numpy(audio).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    feats, _ = wav2vec(wf)
                    pooled = feats.squeeze(0).mean(dim=0).unsqueeze(0)
                    logits = model(pooled)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred = logits.argmax(dim=1).item()
                    
                results.append((label_int, pred, probs[pred].item()))
                status = "OK" if pred == label_int else "FAIL"
                print(f"[{status}] {label_str:4s} | Pred: {pred} | Conf: {probs[pred].item():.2%}")
            except Exception as e:
                print(f"Error processing {f}: {e}")

    if results:
        trues = [r[0] for r in results]
        preds = [r[1] for r in results]
        acc = accuracy_score(trues, preds)
        print(f"\nOverall Verfication Accuracy (Sample): {acc*100:.2f}%")
    else:
        print("No samples processed.")

if __name__ == "__main__":
    verify()
