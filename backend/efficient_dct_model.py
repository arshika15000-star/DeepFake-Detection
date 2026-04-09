import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import cv2
import numpy as np

def extract_freq_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct  = cv2.dct(np.float32(gray))
    # High-frequency quadrant — fakes show unusual energy here
    hf   = dct[gray.shape[0]//2:, gray.shape[1]//2:]
    return np.mean(hf), np.std(hf), np.max(hf)

class EfficientDCTDetector(nn.Module):
    """
    Custom Deepfake Image model that appends 3 DCT High-Frequency values
    directly to the EfficientNet visual feature vector before classification.
    """
    def __init__(self, num_classes=2):
        super(EfficientDCTDetector, self).__init__()
        
        # Load the standard visual backbone
        self.base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        # EfficientNet stores its final fully connected layer at .classifier[-1]
        in_features = self.base_model.classifier[-1].in_features
        
        # Strip the default classifier so it outputs the raw embedding 
        # (typically 1280 dimensions for EfficientNetV2-S)
        self.base_model.classifier[-1] = nn.Identity()
        
        # Build a new classifier that accepts the visual embedding + the 3 DCT features
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, pixel_tensor, dct_tensor):
        """
        Args:
            pixel_tensor: Standard batch of resized images (batch, 3, 224, 224)
            dct_tensor: Batch of DCT features extracted via cv2.dct (batch, 3)
                        [mean, std, max] of the high frequency quadrant.
        Returns:
            Logits for Fake (1)/Real (0) classification
        """
        # Get raw visual features
        visual_features = self.base_model(pixel_tensor) # shape: (batch, in_features)
        
        # Append the frequency-domain (DCT) features
        # shape becomes: (batch, in_features + 3)
        fused_vector = torch.cat((visual_features, dct_tensor), dim=1)
        
        # Run the final multi-layer perceptron (classification head)
        output = self.classifier(fused_vector)
        return output

if __name__ == "__main__":
    # Quick Test to ensure dimensionalities are correct
    model = EfficientDCTDetector()
    
    # Fake batch of 4 images
    dummy_pixels = torch.randn(4, 3, 224, 224) 
    # Fake batch of 4 DCT feature extractions (mean, std, max)
    dummy_dct = torch.randn(4, 3) 
    
    predictions = model(dummy_pixels, dummy_dct)
    print("Architectural Forward Pass Complete.")
    print("Output shape:", predictions.shape) # Should be (4, 2)
