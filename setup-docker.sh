#!/bin/bash

# Set the tag for the Docker image
IMAGE_TAG="yourusername/triton-server:latest"

# Pull the pre-built Docker image from Docker Hub
echo "Pulling the Docker image from Docker Hub..."
docker pull $IMAGE_TAG

# Run the Triton Inference Server container with GPU support
echo "Starting Triton Inference Server..."
docker run --gpus=all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    $IMAGE_TAG tritonserver --model-repository=/triton-inference-server/models

echo "Triton Inference Server is running."

