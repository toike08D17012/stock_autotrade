FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic tools
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    cmake \
    # Image processing dependencies
    libgl1 \
    libglib2.0-0 \
    # Scientific computing dependencies
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    # Utilities
    zip \
    unzip \
    procps \
    nano \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

ARG WORKSPACE=/workspace

ARG USER_UID
ARG USER_GID
ARG USER_NAME
ARG GROUP_NAME=${USER_NAME}

ENV HOME=/home/${USER_NAME}

RUN groupadd --gid ${USER_GID} --non-unique ${GROUP_NAME} \
    && useradd -m --no-log-init --uid ${USER_UID} --non-unique --gid ${USER_GID} --groups sudo --base-dir /home --shell /bin/bash ${USER_NAME} \
    && touch ${HOME}/.sudo_as_admin_successful \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && echo '${USER_NAME} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p ${WORKSPACE} \
    && chown -R ${USER_UID}:${USER_GID} ${WORKSPACE} \
    && chown -R ${USER_UID}:${USER_GID} /home

USER ${USER_NAME}
ARG REPO_NAME
ENV WORK_DIR=${HOME}/workspace/${REPO_NAME}
ENV PYTHON_VERSION=3.14.2 \
    UV_PROJECT=${WORK_DIR}/environments/python \
    UV_PROJECT_ENVIRONMENT=${HOME}/.venv
ENV RUFF_CONFIG=${UV_PROJECT}/pyprojcet.toml

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}:/root/.cargo/bin:${HOME}/.local/bin:${PATH}"


COPY --chown=${USER_UID}:${USER_GID} environments/python/pyproject.toml environments/python/uv.lock ${UV_PROJECT}/

RUN uv python install ${PYTHON_VERSION} && \
    uv venv --python ${PYTHON_VERSION} ${UV_PROJECT_ENVIRONMENT} \
    && uv sync \
    && uv clean

# Default command
CMD ["bash"]
