FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime dependencies only. The challenge executes the final image offline,
# so everything required at inference time must already be installed here.
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "nibabel==5.2.1" \
    "PyYAML==6.0.2"

COPY app/mrixfields /app/mrixfields
COPY app/config.yaml /app/config.yaml
COPY app/inference.py /app/inference.py
COPY app/weights/model.pth /app/weights/model.pth

ENTRYPOINT ["python", "/app/inference.py"]
