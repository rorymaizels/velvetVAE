FROM python:3.10

RUN apt-get update \
    && apt-get install -y git \
    && apt-get install -y curl

RUN useradd -ms /bin/bash dev

USER dev
WORKDIR /home/dev
ENV PATH="/home/dev/.local/bin:${PATH}"