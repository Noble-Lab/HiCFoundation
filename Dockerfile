# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -t -i -p -y

# Set working directory
WORKDIR /app

# Create conda environment and install dependencies directly
RUN conda create -n HiCFoundation python=3.8.10 -y && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "HiCFoundation", "/bin/bash", "-c"]

# Install conda packages
RUN conda install -c pytorch -c nvidia -c conda-forge -c anaconda -c bioconda -c defaults \
    cudatoolkit=11.1.74 \
    pip=21.1.3 \
    pytorch=1.8.1 \
    torchvision=0.9.1 \
    timm=0.3.2 \
    openjdk \
    pandas \
    matplotlib \
    scipy \
    numba \
    cooler \
    -y && \
    conda clean -afy

# Install pip packages
RUN pip install \
    easydict \
    opencv-python \
    simplejson \
    lvis \
    "Pillow==9.5.0" \
    pytorch_msssim  \
    scikit-image \
    einops \
    tensorboard \
    pyBigWig


# Set the default command to activate conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "HiCFoundation"]
CMD ["python", "--version"] 
