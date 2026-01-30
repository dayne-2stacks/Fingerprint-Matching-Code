# Base image with PyTorch + CUDA + PyG
FROM runzhongwang/thinkmatch:torch1.10.0-cuda11.3-cudnn8-pyg2.0.3-pygmtools0.5.3

# Set working directory
WORKDIR /workspace

# Install extra Python deps first (cache-friendly)
COPY requirements.txt /workspace/requirements.txt
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

RUN pip install markupsafe==2.0.1

# Copy rest of project
COPY . /workspace/

# Default command (override when needed)
CMD ["/bin/bash"]
