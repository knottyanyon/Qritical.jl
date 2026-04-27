FROM julia:1.12-bookworm

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


# Set where the repository will live inside the image and copy it in.
WORKDIR /opt/src
COPY . /opt/src

# Make sure docs workdir is explicit
WORKDIR /opt/src/docs

# Activate docs project, develop the package from repo root, instantiate and precompile.
# This assumes:
# - docs/Project.toml exists
# - package Project.toml is at repo root (/opt/src/Project.toml)
# If your package lives in a subdir, change PackageSpec(path=abspath("..")) to the correct path or add subdir.
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.develop(PackageSpec(path=abspath(".."))); Pkg.instantiate(); Pkg.precompile()'

# Keep the working dir for subsequent docs builds
WORKDIR /github/workspace
CMD ["bash"]