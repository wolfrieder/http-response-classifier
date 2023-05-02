# syntax=docker/dockerfile:1
# Use the official Python 3.9.12 image as the base image
FROM python:3.9.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project structure into the container
COPY . /app

# Install Jupyter
RUN pip install jupyter

# Install DVC
RUN pip install dvc

# Set up DVC
RUN dvc init --no-scm --force

# TODO: UPDATE PATHS
# Download the local datasets
COPY data/raw/chrome/08_12_2022/.dvc /app/data/raw/.dvc
COPY data/merged/.dvc /app/data/merged/.dvc

# Add DVC remote storage
RUN dvc remote add -d local_storage /tmp/dvc-storage
RUN dvc config cache.type reflink,symlink,hardlink,copy

# Pull the data from local DVC storage
RUN dvc pull data/raw/.dvc
RUN dvc pull data/merged/.dvc

# Set up the output directory
VOLUME /app/output

# Define the entrypoint to access the terminal within the Docker container
ENTRYPOINT ["/bin/bash"]

# Set port
ENV PORT 8088
EXPOSE $PORT

# Run the pipeline
#CMD ["dvc", "repro", "data/raw/chrome/dvc.yaml"]

# When the pipeline finishes, copy the resulting CSV file to the output directory
#CMD ["cp", "result.csv", "/app/output/result.csv"]

# Set up the Jupyter notebook entrypoint
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
