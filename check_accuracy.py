import torch

ck = torch.load('deepfake_model_best.pth', map_location='cpu')
print('Best model keys:', list(ck.keys()))
print('Epoch saved at  :', ck.get('epoch'))
val_acc = ck.get('val_acc', 0)
if val_acc <= 1.0:
    print(f'Best Val Accuracy: {val_acc * 100:.2f}%')
else:
    print(f'Best Val Accuracy: {val_acc:.2f}%')
