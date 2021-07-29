FROM nvidia/cuda:10.1-base

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6

WORKDIR root/workspace

COPY . .

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
