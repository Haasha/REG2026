FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
# PyTorch + CUDA base so GPU is available at inference time.

# Flush stdout/stderr immediately — prevents missing log lines on container exit.
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Install Python dependencies first (cached layer — only rebuilds when requirements change).
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Copy algorithm source code.
COPY --chown=user:user core.py      /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user src/         /opt/app/src/

ENTRYPOINT ["python", "inference.py"]
