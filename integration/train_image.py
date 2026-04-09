import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score

# Absolute path to this script's directory — ensures dataset is found regardless
# of which directory the script is launched from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # one level up from integration/

# Config
BATCH_SIZE = 32
NUM_EPOCHS = 20           # was 5 — more epochs needed to converge on deepfake patterns
MAX_STEPS_PER_EPOCH = 200  # was 50 — more steps per epoch for meaningful gradient updates
LEARNING_RATE = 3e-4       # was 1e-3 — lower LR needed for fine-tuning pretrained backbone
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(_PROJECT_ROOT, "dataset", "images")
SAVE_DIR = _PROJECT_ROOT

# Cap total samples per class when running on CPU to keep training fast.
# Set to None to use the full dataset (recommended when GPU is available).
MAX_SAMPLES_PER_CLASS = 3000 if not torch.cuda.is_available() else None

_face_detector = None

def face_crop(image):
    """Detect and crop face from PIL image using mediapipe"""
    global _face_detector
    try:
        import mediapipe as mp
        import numpy as np
        import cv2
        from PIL import Image
        
        if _face_detector is None:
            mp_face_detection = mp.solutions.face_detection
            _face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            
        img_np = np.array(image.convert('RGB'))
        results = _face_detector.process(img_np)
        if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                h, w, _ = img_np.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                box_w, box_h = int(bbox.width * w), int(bbox.height * h)
                
                padding = int(max(box_w, box_h) * 0.2)
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(w, x + box_w + padding), min(h, y + box_h + padding)
                
                if x2 > x1 and y2 > y1:
                    crop = img_np[y1:y2, x1:x2]
                    return Image.fromarray(crop)
    except Exception:
        pass
    return image

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Lambda(face_crop),
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(face_crop),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

def build_model(num_classes=2, freeze_backbone=False):
    try:
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)
    except Exception:
        model = models.resnext50_32x4d(pretrained=True)

    if freeze_backbone:
        # Freeze ALL parameters first
        for param in model.parameters():
            param.requires_grad = False
        # Then selectively UNFREEZE the last 2 feature blocks + classifier
        # This gives the model the ability to learn deepfake-specific patterns
        # while keeping early ImageNet features frozen (better generalisation)
        unfreeze_keywords = ['features.6', 'features.7', 'features.8', 'classifier']
        for name, param in model.named_parameters():
            if any(kw in name for kw in unfreeze_keywords):
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] Partial unfreeze: {n_trainable:,} trainable params (last 2 blocks + head)")

    # Replace classification head
    if hasattr(model, 'classifier'):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    return model.to(DEVICE)

def load_datasets(root=DATA_ROOT, val_fraction=0.2):
    if not os.path.exists(root):
        print(f"Image dataset folder not found at {root}. Expected structure: dataset/images/real and dataset/images/fake")
        return None, None

    full = datasets.ImageFolder(root, transform=get_transforms(train=True))

    # Optionally cap samples per class for CPU training
    if MAX_SAMPLES_PER_CLASS is not None:
        from collections import defaultdict
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full.samples):
            class_indices[label].append(idx)

        capped_indices = []
        for label, indices in class_indices.items():
            capped_indices.extend(indices[:MAX_SAMPLES_PER_CLASS])

        full = torch.utils.data.Subset(full, capped_indices)
        print(f"[CPU Mode] Capped to {MAX_SAMPLES_PER_CLASS} samples/class -> {len(full)} total images")
    else:
        print(f"[Training] Found {len(full)} total images")

    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader

def train():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = load_datasets()
    if train_loader is None:
        return

    # Partially unfreeze backbone (last 2 blocks) — critical for learning deepfake features
    model = build_model(freeze_backbone=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    scaler = torch.amp.GradScaler() if DEVICE.type == 'cuda' else None

    best_acc = 0.0
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        preds = []
        trues = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= MAX_STEPS_PER_EPOCH:
                break
                
            if batch_idx > 0 and batch_idx % 10 == 0:
                print(f"  -> Batch {batch_idx}/{min(len(train_loader), MAX_STEPS_PER_EPOCH)} - Loss: {loss.item():.4f}")

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

        train_acc = accuracy_score(trues, preds) if trues else 0.0

        # Validation
        model.eval()
        v_preds = []
        v_trues = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if batch_idx >= MAX_STEPS_PER_EPOCH // 2: # Evaluate fewer batches to save time
                    break
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                v_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
                v_trues.extend(labels.cpu().numpy().tolist())

        val_acc = accuracy_score(v_trues, v_preds) if v_trues else 0.0
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {running_loss:.4f}")

        ck = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }
        torch.save(ck, f"deepfake_model_epoch_{epoch+1}.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ck, "deepfake_model_best.pth")

    torch.save(model.state_dict(), "deepfake_model_final.pth")
    total = time.time() - start
    print(f"Image training completed in {total:.2f}s. Best val acc: {best_acc:.4f}. Final model saved to deepfake_model_final.pth")

if __name__ == '__main__':
    train()
