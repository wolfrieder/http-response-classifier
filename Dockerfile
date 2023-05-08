# syntax=docker/dockerfile:1
# Use the official Python 3.9.12 image as the base image
FROM arm64v8/ubuntu
#FROM arm64v8/python:3.10.11-slim-bullseye
#FROM python:3.10.11-slim

# Set the working directory
WORKDIR /app

# Install required system packages for numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils python3-pip

# Copy the requirements.txt file into the container
COPY requirements_docker.txt /app

# Create and activate a virtual environment
RUN python3.10 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip
# Install the required packages
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the project structure into the container
COPY . /app

# TODO: UPDATE PATHS
# Download the local datasets
#COPY data/raw/chrome/08_12_2022/.dvc /app/data/raw/.dvc
#COPY data/merged/.dvc /app/data/merged/.dvc

# Add DVC remote storage
#RUN dvc remote add -d local_storage /tmp/dvc-storage
#RUN dvc config cache.type reflink,symlink,hardlink,copy

# Pull the data from local DVC storage
#RUN dvc pull data/raw/.dvc
#RUN dvc pull data/merged/.dvc

# Set up the output directory
VOLUME /app/output

# Define the entrypoint to access the terminal within the Docker container
ENTRYPOINT ["/bin/bash"]

# Set port
ENV PORT 8088
EXPOSE $PORT

# Run the pipeline
CMD ["dvc", "repro", "data/raw/chrome/dvc.yaml"]

# When the pipeline finishes, copy the resulting CSV file to the output directory
#CMD ["cp", "result.csv", "/app/output/result.csv"]

# Set up the Jupyter notebook entrypoint
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

