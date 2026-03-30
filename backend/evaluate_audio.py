import os
import sys
import torch
import torchaudio
import glob
import random
import time

def evaluate_audio_model(data_dir, max_files=200):
    print("==================================================")
    print(" DeepTruth AI - Audio Model Accuracy Evaluation")
    print(f" Mode: LIGHTWEIGHT (max {max_files} files sampled)")
    print("==================================================")

    # ── Throttle CPU so laptop stays responsive ──────────────────────────────
    torch.set_num_threads(2)          # only use 2 of your CPU cores
    torch.set_num_interop_threads(1)

    from text_audio_models import AudioDiscriminator

    device = torch.device('cpu')     # always CPU — avoids memory spikes
    print(f"Using device: {device} (throttled to 2 threads)")

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
        print("WARNING: No trained weights found — results will be random baseline.")
        print("  Searched in:", _script_dir, "and", data_dir)

    model.eval()

    # Load Wav2Vec2
    print("Loading Torchaudio Wav2Vec2 feature extractor...")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec_model = bundle.get_model().to(device)
    wav2vec_model.eval()

    real_files = glob.glob(os.path.join(data_dir, "real", "*.wav"))
    fake_files = glob.glob(os.path.join(data_dir, "fake", "*.wav"))

    if not real_files and not fake_files:
        print(f"\nNo audio files found in {data_dir}/real or {data_dir}/fake!")
        print("Please ensure your dataset is structured correctly.")
        return

    # ── Balanced random sample (half real, half fake) ────────────────────────
    half = max_files // 2
    sampled_real = random.sample(real_files, min(half, len(real_files)))
    sampled_fake = random.sample(fake_files, min(half, len(fake_files)))
    all_files = [(f, 1) for f in sampled_real] + [(f, 0) for f in sampled_fake]
    random.shuffle(all_files)

    total_avail = len(real_files) + len(fake_files)
    print(f"\nDataset: {len(real_files)} real + {len(fake_files)} fake = {total_avail} total")
    print(f"Evaluating a random balanced sample of {len(all_files)} files...")
    print("(Your laptop stays responsive — press Ctrl+C to stop early)\n")

    correct = 0
    errors  = 0
    total   = len(all_files)

    for i, (file_path, label) in enumerate(all_files, 1):
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(file_path)
            if audio.ndim == 1:
                waveform = torch.from_numpy(audio).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(audio[:, 0]).unsqueeze(0).float()

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Trim to max 5 seconds to save time
            max_samples = 16000 * 5
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

            with torch.no_grad():
                features, _ = wav2vec_model(waveform)
                pooled = features.mean(dim=1)      # [1, 768]
                logits = model(pooled).squeeze(0)
                probs  = torch.softmax(logits, dim=0)
                pred_label = 1 if probs[1] > 0.5 else 0

            if pred_label == label:
                correct += 1

        except KeyboardInterrupt:
            print("\n[Stopped early by user]")
            total = i - 1
            break
        except Exception as e:
            errors += 1
            total -= 1

        # ── Progress update every 10 files ──
        if i % 10 == 0 or i == len(all_files):
            running_acc = (correct / max(i - errors, 1)) * 100
            print(f"  [{i:>3}/{len(all_files)}]  Running accuracy: {running_acc:.1f}%  |  Errors: {errors}")
            sys.stdout.flush()

        # ── Small sleep to keep laptop responsive ──
        time.sleep(0.05)   # 50ms pause between files

    print()
    print("==================================================")
    if total > 0:
        accuracy = (correct / total) * 100
        print(f" AUDIO EVALUATION COMPLETE")
        print(f" Sampled {total} files from {total_avail} total")
        print(f" ACCURACY: {accuracy:.2f}%  ({correct}/{total} correct)")
        print(f" Errors skipped: {errors}")
    else:
        print(" No files could be evaluated.")
    print("==================================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate audio model (laptop-friendly)")
    parser.add_argument("--data_dir",  type=str, default="../dataset/audio",
                        help="Directory with 'real' and 'fake' subfolders")
    parser.add_argument("--max_files", type=int, default=200,
                        help="Max files to evaluate (default 200, ≈5 min on CPU)")
    args = parser.parse_args()
    evaluate_audio_model(args.data_dir, max_files=args.max_files)
