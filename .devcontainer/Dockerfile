FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary dependencies.
RUN apt-get update && apt-get install -y \
    gfortran \
    build-essential \
    wget \
    curl \
    libblas3 \
    libblas-dev \
    liblapack3 \
    liblapack-dev \
    libmetis-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Julia.
ARG JULIA_VERSION=1.11.3
RUN wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia \
    && rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

# Add user.
ARG USER_NAME=developer
ARG USER_ID=1001
ARG USER_HOME="/home/$USER_NAME"

RUN adduser \
    --home "$USER_HOME" \
    --uid $USER_ID \
    --disabled-password \
    "$USER_NAME"

# Set the default user.
USER ${USER_NAME}

# Set the working directory to the user's home directory.
WORKDIR ${USER_HOME}

# Create a Julia package depot
ENV JULIA_DEPOT_PATH="/home/developer/.julia"

# Copy Project.toml and Manifest.toml, copy and extract HSL_jll.jl, and install all packages.
COPY --chown=${USER_NAME}:${USER_NAME} Project.toml Manifest.toml ${USER_HOME}/
COPY --chown=${USER_NAME}:${USER_NAME} .devcontainer/HSL_jll.jl.v2024.11.28.tar.gz ${USER_HOME}/
RUN tar -xzvf HSL_jll.jl.v2024.11.28.tar.gz \
    && julia -e 'using Pkg; Pkg.activate(); Pkg.develop(path="HSL_jll.jl.v2024.11.28", shared=true, preserve=Pkg.PRESERVE_TIERED); Pkg.instantiate(); Pkg.activate(.); Pkg.instantiate()' \
    && rm HSL_jll.jl.v2024.11.28.tar.gz \
    && rm HSL_jll.pdf\
    && rm build_tarballs.jl \
    && rm Project.toml \
    && rm Manifest.toml