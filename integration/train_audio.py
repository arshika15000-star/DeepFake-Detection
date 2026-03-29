import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm

sys.path.append('../backend')
from text_audio_models import AudioDiscriminator

# Settings
DATA_DIR = os.path.join("..", "dataset", "audio")
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioDataset(Dataset):
    def __init__(self, data_dir, wav2vec_model):
        self.files = []
        self.labels = []
        self.wav2vec_model = wav2vec_model
        
        real_files = glob(os.path.join(data_dir, "real", "*.wav"))
        fake_files = glob(os.path.join(data_dir, "fake", "*.wav"))
        
        for f in real_files:
            self.files.append(f)
            self.labels.append(1)  # Real
            
        for f in fake_files:
            self.files.append(f)
            self.labels.append(0)  # Fake
            
        print(f"Loaded {len(real_files)} real and {len(fake_files)} fake audio files.")
            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(file_path)
            # Handle PyTorch shape requirements (channels, frames)
            if audio.ndim == 1:
                waveform = torch.from_numpy(audio).unsqueeze(0).float()
            else:
                waveform = torch.from_numpy(audio).transpose(0, 1).float()
                
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            with torch.no_grad():
                waveform = waveform.to(device)
                features, _ = self.wav2vec_model(waveform)
                features = features.squeeze(0).mean(dim=0).cpu() # Average over time to get 1D feature
        except Exception as e:
            # Fallback for corrupted files
            features = torch.zeros(768)
            print(f"Error on {file_path}: {e}")
            
        return features, torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    print("====================================")
    print(" DeepTruth AI - Audio Training")
    print("====================================")

    if not os.path.exists(os.path.join(DATA_DIR, "real")) or not os.path.exists(os.path.join(DATA_DIR, "fake")):
        print("Dataset not found! Please run setup_audio_dataset.py first.")
        sys.exit(1)

    print("Loading Torchaudio Wav2Vec2.0 feature extractor (offline proxy)...")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec_model = bundle.get_model().to(device)
    wav2vec_model.eval()

    print("Preparing Audio Dataset...")
    dataset = AudioDataset(DATA_DIR, wav2vec_model)
    
    if len(dataset) == 0:
        print("No audio data found! Exiting.")
        sys.exit(1)
        
    # Standard 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training set: {train_size} | Validation set: {val_size}")

    model = AudioDiscriminator(feature_dim=768).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_acc = 0.0

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                v_total += labels.size(0)
                v_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * v_correct / v_total
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f" ---> Best Model Saved! Accuracy: {best_val_acc:.2f}%")
            torch.save(model.state_dict(), "../backend/audio_model_best.pth")

    print("\nTraining Complete! Saving final weights.")
    torch.save(model.state_dict(), "../backend/audio_model_final.pth")
    print("Done. To use this accuracy in the app, the backend has been updated to automatically load 'audio_model_final.pth'.")
