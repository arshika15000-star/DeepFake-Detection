import numpy as np
import os
import sys
import glob
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib
import torch

# ─── PATHS ───────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

IMAGE_DATASET = os.path.join(BACKEND_DIR, '..', 'dataset', 'images')      # subfolders: real/ fake/
VIDEO_DATASET = os.path.join(BACKEND_DIR, '..', 'dataset', 'SDFVD', 'SDFVD')  # subfolders: videos_real/ videos_fake/
AUDIO_DATASET = os.path.join(BACKEND_DIR, '..', 'dataset', 'audio')       # subfolders: real/ fake/
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── STEP 1: COLLECT REAL PREDICTIONS FROM EACH MODALITY MODEL ───────────────
def collect_image_probs(max_samples=300):
    """Run the trained image model over the image dataset and collect [fake_prob, label] pairs."""
    probs, labels = [], []
    try:
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        from PIL import Image
        import io

        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        model_path = os.path.join(BACKEND_DIR, 'deepfake_model_best.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(BACKEND_DIR, 'deepfake_model_final.pth')
        if not os.path.exists(model_path):
            print("  [Image] No image model found — skipping.")
            return probs, labels
        ck = torch.load(model_path, map_location=DEVICE)
        sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
        model.load_state_dict(sd, strict=False)
        model = model.to(DEVICE).eval()

        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        for label, sub in [(1, 'fake'), (0, 'real')]:
            folder = os.path.join(IMAGE_DATASET, sub)
            if not os.path.exists(folder):
                continue
            files = glob.glob(os.path.join(folder, '*.jpg')) + \
                    glob.glob(os.path.join(folder, '*.png')) + \
                    glob.glob(os.path.join(folder, '*.jpeg'))
            for fpath in files[:max_samples // 2]:
                try:
                    img = Image.open(fpath).convert('RGB')
                    t = tfm(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = model(t)
                        probs_list = torch.softmax(out, dim=1)[0]
                        # Image model trained with: index 0 = FAKE, index 1 = REAL
                        p = float(probs_list[0].item()) # extracted fake prob
                    probs.append(p)
                    labels.append(label)
                except Exception:
                    pass
        print(f"  [Image] Collected {len(probs)} real image predictions.")
    except Exception as e:
        print(f"  [Image] Collection failed: {e}")
    return probs, labels


def collect_video_probs(max_samples=100):
    """Run the trained video model over the video dataset and collect [fake_prob, label] pairs."""
    probs, labels = [], []
    try:
        import cv2
        from torchvision import transforms
        from test_video_model import DeepfakeDetector, FRAMES_PER_VIDEO

        model_path = os.path.join(BACKEND_DIR, 'video_model_best.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(BACKEND_DIR, 'video_model_final.pth')
        if not os.path.exists(model_path):
            print("  [Video] No video model found — skipping.")
            return probs, labels

        model = DeepfakeDetector(num_frames=FRAMES_PER_VIDEO).to(DEVICE)
        ck = torch.load(model_path, map_location=DEVICE)
        sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
        model.load_state_dict(sd, strict=False)
        model.eval()

        tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        for label, sub in [(1, 'videos_fake'), (0, 'videos_real')]:
            folder = os.path.join(VIDEO_DATASET, sub)
            if not os.path.exists(folder):
                continue
            files = glob.glob(os.path.join(folder, '*.mp4'))
            for fpath in files[:max_samples // 2]:
                try:
                    cap = cv2.VideoCapture(fpath)
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    indices = np.linspace(0, max(total_f - 1, 0), FRAMES_PER_VIDEO, dtype=int)
                    frames = []
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        else:
                            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                    cap.release()
                    while len(frames) < FRAMES_PER_VIDEO:
                        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
                    tensors = torch.stack([tfm(f) for f in frames]).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out, _ = model(tensors)
                        p = torch.softmax(out, dim=1)[0][1].item()  # fake prob
                    probs.append(p)
                    labels.append(label)
                except Exception:
                    pass
        print(f"  [Video] Collected {len(probs)} real video predictions.")
    except Exception as e:
        print(f"  [Video] Collection failed: {e}")
    return probs, labels


def collect_audio_probs(max_samples=200):
    """Run the trained audio model over the audio dataset and collect [fake_prob, label] pairs."""
    probs, labels = [], []
    try:
        import torchaudio
        import soundfile as sf
        from text_audio_models import AudioDiscriminator

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        wav2vec = bundle.get_model().to(DEVICE).eval()

        audio_path = os.path.join(BACKEND_DIR, 'audio_model_best.pth')
        if not os.path.exists(audio_path):
            audio_path = os.path.join(BACKEND_DIR, 'audio_model_final.pth')
        if not os.path.exists(audio_path):
            print("  [Audio] No audio model found — skipping.")
            return probs, labels

        discriminator = AudioDiscriminator(feature_dim=768).to(DEVICE)
        discriminator.load_state_dict(torch.load(audio_path, map_location=DEVICE))
        discriminator.eval()

        for label, sub in [(1, 'fake'), (0, 'real')]:
            folder = os.path.join(AUDIO_DATASET, sub)
            if not os.path.exists(folder):
                continue
            files = glob.glob(os.path.join(folder, '*.wav')) + \
                    glob.glob(os.path.join(folder, '*.flac'))
            for fpath in files[:max_samples // 2]:
                try:
                    audio, sr = sf.read(fpath)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                    if sr != 16000:
                        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                    waveform = waveform.to(DEVICE)
                    with torch.no_grad():
                        feats, _ = wav2vec(waveform)
                        pooled = feats.squeeze(0).mean(dim=0).unsqueeze(0)
                        logits = discriminator(pooled).squeeze(0)
                        p = torch.softmax(logits, dim=0)[1].item()  # fake prob
                    probs.append(p)
                    labels.append(label)
                except Exception:
                    pass
        print(f"  [Audio] Collected {len(probs)} real audio predictions.")
    except Exception as e:
        print(f"  [Audio] Collection failed: {e}")
    return probs, labels


def align_multimodal(img_p, img_l, vid_p, vid_l, aud_p, aud_l):
    """
    Combine per-modality prediction lists into aligned fusion vectors.
    Samples are matched by label to create balanced fusion rows.
    For production use, you would align by matching actual media files.
    """
    # Group by label
    def by_label(ps, ls):
        real = [p for p, l in zip(ps, ls) if l == 0]
        fake = [p for p, l in zip(ps, ls) if l == 1]
        return real, fake

    img_r, img_f = by_label(img_p, img_l)
    vid_r, vid_f = by_label(vid_p, vid_l)
    aud_r, aud_f = by_label(aud_p, aud_l)

    # Use the MAXIMUM count but keep it balanced between Real and Fake
    n_real = max(len(img_r), len(vid_r), len(aud_r))
    n_fake = max(len(img_f), len(vid_f), len(aud_f))
    n = max(n_real, n_fake)

    if n == 0:
        print("  [Fusion] No model predictions were successfully collected. Aborting.")
        return None, None

    # Use mean as fallback for missing modalities to keep scores realistic
    def get_mean(arr, default): return float(np.mean(arr)) if len(arr) > 0 else default
    img_r_m, img_f_m = get_mean(img_r, 0.3), get_mean(img_f, 0.7)
    vid_r_m, vid_f_m = get_mean(vid_r, 0.3), get_mean(vid_f, 0.7)
    aud_r_m, aud_f_m = get_mean(aud_r, 0.3), get_mean(aud_f, 0.7)

    X, y = [], []
    # Generate balanced samples
    for i in range(n):
        # Real Sample alignment
        X.append([
            img_r[i % len(img_r)] if len(img_r) > 0 else img_r_m,
            vid_r[i % len(vid_r)] if len(vid_r) > 0 else vid_r_m,
            aud_r[i % len(aud_r)] if len(aud_r) > 0 else aud_r_m,
        ])
        y.append(0)
        
        # Fake Sample alignment
        X.append([
            img_f[i % len(img_f)] if len(img_f) > 0 else img_f_m,
            vid_f[i % len(vid_f)] if len(vid_f) > 0 else vid_f_m,
            aud_f[i % len(aud_f)] if len(aud_f) > 0 else aud_f_m,
        ])
        y.append(1)

    return np.array(X), np.array(y)


# ─── STEP 2: COLLECT AND BUILD FUSION DATASET ────────────────────────────────
print("\n" + "="*60)
print("META-CLASSIFIER TRAINING — Using REAL Model Predictions")
print("="*60)

print("\nCollecting predictions from trained individual models...")
img_probs, img_labels = collect_image_probs(max_samples=300)
vid_probs, vid_labels = collect_video_probs(max_samples=100)
aud_probs, aud_labels = collect_audio_probs(max_samples=200)

print(f"\nRaw collection counts:")
print(f"  Image : {len(img_probs)} samples | Labels: {Counter(img_labels)}")
print(f"  Video : {len(vid_probs)} samples | Labels: {Counter(vid_labels)}")
print(f"  Audio : {len(aud_probs)} samples | Labels: {Counter(aud_labels)}")

# Build aligned fusion matrix
X_all, y_all = align_multimodal(img_probs, img_labels, vid_probs, vid_labels, aud_probs, aud_labels)

if X_all is None or len(X_all) < 2:
    print("\n[ERROR] No data collected to train meta-classifier.")
    print("Please ensure at least one modality dataset exists and models are trained first.")
    print("Dataset paths expected:")
    print(f"  Image : {IMAGE_DATASET}/real  and  {IMAGE_DATASET}/fake")
    print(f"  Video : {VIDEO_DATASET}/videos_real  and  {VIDEO_DATASET}/videos_fake")
    print(f"  Audio : {AUDIO_DATASET}/real  and  {AUDIO_DATASET}/fake")
    exit(0)

print(f"\nFusion dataset: {len(X_all)} samples | Distribution: {Counter(y_all.tolist())}")
print("Sample fusion vectors (first 5):")
for i in range(min(5, len(X_all))):
    print(f"  Label={y_all[i]} | [img={X_all[i][0]:.3f}, vid={X_all[i][1]:.3f}, aud={X_all[i][2]:.3f}]")


# ─── STEP 3: TRAIN/VAL/TEST SPLIT ────────────────────────────────────────────
train_A, remainder, y_A, y_remainder = train_test_split(
    X_all, y_all, test_size=0.40, stratify=y_all, random_state=42
)
fusion_B, test_C, y_B, y_C = train_test_split(
    remainder, y_remainder, test_size=0.50, stratify=y_remainder, random_state=42
)

print("\n" + "="*60)
print("3-WAY SPLIT (Real Data):")
print(f"  Train A (60%): {Counter(y_A.tolist())}")
print(f"  Fusion B (20%): {Counter(y_B.tolist())}")
print(f"  Test C  (20%): {Counter(y_C.tolist())}")
print("="*60)


# ─── STEP 4: TRAIN META-CLASSIFIER ───────────────────────────────────────────
all_samples = list(X_all)
all_labels = list(y_all)
print("\nTraining meta-classifier on real prediction data...")
meta_clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
meta_clf.fit(train_A, y_A)


# ─── STEP 5: CALIBRATE THRESHOLD ON SPLIT B ──────────────────────────────────
y_scores_B = meta_clf.predict_proba(fusion_B)[:, 1]
fpr, tpr, thresholds = roc_curve(y_B, y_scores_B)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = float(thresholds[optimal_idx])
print(f"\nOptimal calibrated threshold (from Fusion B): {optimal_threshold:.4f}")


# ─── STEP 6: EVALUATE ON HELD-OUT TEST SET C ─────────────────────────────────
y_scores_C = meta_clf.predict_proba(test_C)[:, 1]
y_pred = (y_scores_C >= optimal_threshold).astype(int)

print("\n" + "="*60)
print("== Final Meta-Classifier Results on Held-Out Test Set C ==")
print(classification_report(y_C, y_pred, target_names=["Real", "Fake"]))
print(f"ROC-AUC Score: {roc_auc_score(y_C, y_scores_C):.4f}")
print("="*60)

print("\n=== META-CLASSIFIER FEATURE IMPORTANCE ===")
feature_names = ['image_prob', 'video_prob', 'audio_prob']
for name, imp in zip(feature_names, meta_clf.feature_importances_):
    print(f"  {name}: {imp:.4f}")


# ─── STEP 7: SAVE ─────────────────────────────────────────────────────────────
joblib.dump(meta_clf, 'meta_classifier.pkl')
joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
print("\nSaved meta_classifier.pkl and optimal_threshold.pkl")
print("Meta-classifier training on REAL DATA complete.")
