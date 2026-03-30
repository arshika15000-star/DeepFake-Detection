import os
import torch
import torchaudio
import glob

def evaluate_audio_model(data_dir):
    print("==================================================")
    print(" DeepTruth AI - Audio Model Accuracy Evaluation")
    print("==================================================")
    
    # 1. Load Audio Discriminator (if it exists)
    from text_audio_models import AudioDiscriminator
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    hidden_dim = 768
    model = AudioDiscriminator(feature_dim=hidden_dim).to(device)
    
    # Check for weights — try backend dir first, then data_dir
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = None
    for candidate in [
        os.path.join(_script_dir, "audio_model_best.pth"),
        os.path.join(_script_dir, "audio_model_final.pth"),
        os.path.join(data_dir,    "audio_model_best.pth"),
        os.path.join(data_dir,    "audio_model_final.pth"),
    ]:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path:
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No trained weights found. Results will be random baseline.")
        print("Searched in:", _script_dir, "and", data_dir)
    
    model.eval()

    # Load Wav2Vec2 Feature Extractor as frontend baseline
    print("Loading Torchaudio Wav2Vec2 feature extractor...")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec_model = bundle.get_model().to(device)
    wav2vec_model.eval()
    
    real_files = glob.glob(os.path.join(data_dir, "real", "*.wav"))
    fake_files = glob.glob(os.path.join(data_dir, "fake", "*.wav"))
    
    if not real_files and not fake_files:
        print(f"\nNo audio files found in {data_dir}/real or {data_dir}/fake!")
        print("Please ensure your dataset is structured correctly to benchmark accuracy.")
        return

    all_files = [(f, 1) for f in real_files] + [(f, 0) for f in fake_files]
    
    correct = 0
    total = len(all_files)
    
    print(f"\nFound {total} files. Evaluating...")
    for file_path, label in all_files:
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(file_path)
            if audio.ndim == 1:
                waveform = torch.from_numpy(audio).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(audio).transpose(0, 1).float()
                
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            waveform = waveform.to(device)
            with torch.no_grad():
                features, _ = wav2vec_model(waveform)
                # the AudioDiscriminator expects a certain shape. Adjust as per text_audio_models.py
                logits = model(features).squeeze(0)
                probs = torch.softmax(logits, dim=0)
                pred_label = 1 if probs[1] > 0.5 else 0
                
            if pred_label == label:
                correct += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            total -= 1
            
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n==================================================")
        print(f" AUDIO EVALUATION COMPLETE ")
        print(f" TOTAL ACCURACY: {accuracy:.1f}% ({correct}/{total})")
        print(f"==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate audio model accuracy")
    parser.add_argument("--data_dir", type=str, default="../dataset/audio", help="Directory containing 'real' and 'fake' subfolders")
    args = parser.parse_args()
    evaluate_audio_model(args.data_dir)
