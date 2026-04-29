import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torchaudio
from transformers import pipeline

def main():
    print("Downloading EfficientNet...")
    _ = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    print("Downloading Wav2Vec2...")
    _ = torchaudio.pipelines.WAV2VEC2_BASE.get_model()

    print("Downloading Text Model...")
    _ = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")

    print("All HuggingFace & PyTorch models downloaded and cached!")

if __name__ == "__main__":
    main()
