import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import librosa
import numpy as np

def extract_audio_features(audio, sr=16000):
    """
    Enhanced feature extraction for forensic audio analysis.
    Combines MFCCs with spectral descriptors to capture AI-generated 'mechanical' signatures.
    """
    # 1. MFCCs (Standard)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 2. Chroma (Harmonic content) - AI often struggles with complex harmonic alignment
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # 3. Spectral Contrast - Captures the 'clarity' vs 'muffleness' of audio bands
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
    # Concatenate all features
    # MFCCs: 120, Chroma: 12, Contrast: 7 -> Total: 139 features per frame
    features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2, chroma, contrast], axis=0)
    return features

# Text Discriminator (GAN Discriminator style architecture)
# =========================================================
class TextDiscriminator(nn.Module):
    """
    A Neural Discriminator to classify text as Human (Real) or AI (Fake).
    Uses an Embedding layer followed by a Transformer Encoder and a classification head.
    """
    def __init__(self, vocab_size=5000, embed_size=128, num_heads=4, hidden_dim=256, num_layers=2):
        super(TextDiscriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, embed_size))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fake vs Real discriminator head
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        seq_len = x.size(1)
        
        # Embed and add positional encoding
        embedded = self.embedding(x)
        embedded = embedded + self.pos_encoder[:, :seq_len, :]
        
        # Pass through transformer
        transformed = self.transformer(embedded)
        
        # Global Average Pooling (summarize the text context)
        pooled = transformed.mean(dim=1)
        
        # Classify (Logits for Fake/Real)
        logits = self.classifier(pooled)
        return logits

def tokenize_text(text, max_len=100, vocab_size=5000):
    """A simplistic fallback tokenizer that hashes words to integer IDs"""
    words = text.lower().replace('.', ' ').replace(',', ' ').split()
    # Simple hash-based tokenization to bypass needing a real tokenizer dictionary
    tokens = [(abs(hash(w)) % (vocab_size - 1)) + 1 for w in words]
    
    # Pad or truncate
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [0] * (max_len - len(tokens))
    
    return torch.tensor([tokens], dtype=torch.long)

# =========================================================
# Audio Discriminator (GAN Discriminator style architecture)
# =========================================================
class AudioDiscriminator(nn.Module):
    """
    A Neural Discriminator to classify audio features.
    It takes the extracted features from Wav2Vec and predicts Human (Real) vs Synthetic (Fake).
    """
    def __init__(self, feature_dim=768):  # 768 is default Wav2Vec2 hidden dim
        super(AudioDiscriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 2)
        )

    def forward(self, wav2vec_features):
        # Pool the wav2vec temporal features if they are 3D
        if len(wav2vec_features.shape) == 3:
            wav2vec_features = wav2vec_features.mean(dim=1)
            
        logits = self.discriminator(wav2vec_features)
        return logits
