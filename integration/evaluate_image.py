import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from train_image import build_model, IMAGE_SIZE, DEVICE

DATA_ROOT = os.path.join('dataset', 'images')

def face_crop(image):
    """Detect and crop face from image using mediapipe"""
    try:
        import mediapipe as mp
        import numpy as np
        import cv2
        
        # Convert PIL to numpy
        img_np = np.array(image)
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(img_np)
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

def get_transform():
    return transforms.Compose([
        transforms.Lambda(face_crop),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def load_loader(batch_size=32):
    ds = ImageFolder(DATA_ROOT, transform=get_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader

def evaluate(model_path='deepfake_image_best.pth'):
    print(f'Loading image model from {model_path} on {DEVICE}')
    model = build_model()
    if not os.path.exists(model_path):
        print('Model file not found:', model_path)
        return
    ck = torch.load(model_path, map_location=DEVICE)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        model.load_state_dict(ck['model_state_dict'])
    else:
        model.load_state_dict(ck)

    model.eval()
    loader = load_loader()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            scores = probs[:,1].cpu().numpy().tolist()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds)
            y_score.extend(scores)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')

    print('Image Evaluation results:')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1-score:  {f1:.4f}')
    print(f'  ROC AUC:   {auc:.4f}')

if __name__ == "__main__":
    evaluate()
