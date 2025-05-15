FROM python:3.11-slim

WORKDIR /workspace

# Installiere Entwicklungstools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installiere TensorFlow und andere Abhängigkeiten
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    numpy==1.24.2 \
    pillow \
    scipy \
    scikit-learn \
    pyheif \
    matplotlib

# Setze den pythonPath auf den Standard-Python-Interpreter
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Setze eine gute Standard-Shell
SHELL ["/bin/bash", "-c"]