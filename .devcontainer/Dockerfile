FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    gcc \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

WORKDIR /workspace

USER $USERNAME
ENV PATH="/home/vscode/.local/bin:${PATH}"