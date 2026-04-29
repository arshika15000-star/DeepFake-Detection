FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, mediapipe, audio processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user that Hugging Face requires
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy requirements file first to leverage Docker cache
COPY --chown=user requirements.txt .

# Install dependencies
# Using --extra-index-url is crucial here! It forces pip to download the much smaller 
# CPU-only versions of PyTorch instead of the massive GPU versions. This keeps your 
# Docker image small and cheap to host.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download huggingface and PyTorch models to bake them into the Docker image
COPY --chown=user download_models.py .
RUN python download_models.py

# Copy the rest of the backend source code
COPY --chown=user . .

# Expose the API port expected by Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI server via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
