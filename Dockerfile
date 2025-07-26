# Use a CUDA-compatible PyTorch image
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Default command
CMD ["python", "train.py"]
