FROM julia:1.10-bullseye

# Put package depot in a stable location inside the image
ENV JULIA_DEPOT_PATH=/opt/julia_depot

# Install system dependencies commonly needed by CairoMakie and similar packages.
# Adjust these if a package requires additional system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    curl \
    wget \
    pkg-config \
    libcairo2-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/julia_env

# Copy docs Project + Manifest (Manifest is recommended for reproducible install).
# Make sure docs/Manifest.toml is committed; if not available the image will still work but
# the instantiate step will resolve and download package versions at build time.
COPY docs/Project.toml ./
# COPY docs/Project.toml docs/Manifest.toml ./

# Instantiate and precompile the docs environment at image build time.
# Using --project=. ensures the docs environment is used.
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile()'

# Keep the working dir for subsequent docs builds
WORKDIR /github/workspace
CMD ["bash"]
