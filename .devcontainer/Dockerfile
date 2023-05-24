FROM python:3.10

RUN apt-get update \
    && apt-get install -y git \
    && apt-get install -y curl \
    && apt-get install -y build-essential \
    && apt-get install -y cmake \
    && apt-get install -y libopenblas-dev \
    && apt-get install -y gfortran \
    && pip install torch torchvision torchaudio

RUN useradd -ms /bin/bash dev

USER dev
WORKDIR /home/dev
ENV PATH="/home/dev/.local/bin:${PATH}"