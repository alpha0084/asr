# Reproducible GPU env for Malayalam-audio -> English fine-tuning + eval.
# Base = torch 2.7.1 + CUDA 12.8 (REQUIRED for the RTX 5090 / Blackwell sm_120; older
# torch has "no kernel image available for execution on the device").
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Code is mounted at /work, data at /data (set ASR_FT_SRC=/data), HF cache at /root/.cache/huggingface.
# Build:  docker build -f finetune/Dockerfile -t asr-finetune finetune
# Run:    docker run --rm -it --gpus all -v %cd%:/work -v D:\project\ft-data:/data \
#           -v D:\project\ft-hfcache:/root/.cache/huggingface -e HF_TOKEN=... -e ASR_FT_SRC=/data asr-finetune bash
CMD ["bash"]
