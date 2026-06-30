# Base Image: PyTorch with CUDA 12.1 + cuDNN 8
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables to optimize PyTorch/CUDA execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    # Optional: HuggingFace hub cache directory
    HF_HOME=/app/cache/huggingface

WORKDIR /app

# Install system dependencies (if any are needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server using Uvicorn
# Note: For production, we bind to 0.0.0.0 so K8s can route traffic to it.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
