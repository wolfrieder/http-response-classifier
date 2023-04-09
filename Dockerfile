# syntax=docker/dockerfile:1
FROM python:3.9.12-slim-bullseye
RUN apt update
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install conda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh && \
    sh Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p /opt/conda && \
    rm -f Miniconda3-py39_4.10.3-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Create a conda environment and install required packages
RUN conda create -y -n ml_env python=3.9.12 && \
    conda activate ml_env && \
    pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY data /app/data

ENV PORT 8088
EXPOSE $PORT

CMD [ "/bin/bash", "-c", "conda activate ml_env && python src/main.py" ]