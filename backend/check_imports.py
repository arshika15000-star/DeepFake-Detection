import sys
try:
    import torch
    import torchvision
    import fastapi
    import uvicorn
    import cv2
    import numpy as np
    import PIL
    import mediapipe
    print("CORE_IMPORT_OK")
except Exception as e:
    print(f"CORE_IMPORT_ERROR: {e}")
