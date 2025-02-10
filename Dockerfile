FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
WORKDIR /usr/app/
COPY . /usr/app/
# ✅ Install all required dependencies before installing Python packages
RUN apt-get update && apt-get install -y \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    libsm6 \
    libxrender-dev \
    libxext6 \
    ffmpeg \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools \
    gstreamer1.0-libav

# Upgrade pip
RUN pip install pip -U
# Install PyTorch and dependencies
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
RUN pip install torchao --index-url https://download.pytorch.org/whl/nightly/cu124
# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install --no-deps facenet_pytorch==2.6.0
RUN pip install fastapi uvicorn aiohttp

# Set environment variables for CUDA & cuDNN
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

EXPOSE 8000
# Run FastAPI server

CMD ["python", "api.py"]