#!/bin/bash

# Check if the environment variable MODEL_REPOSITORY is defined
if [ -z "$MODEL_REPOSITORY" ]; then
  echo "MODEL_REPOSITORY is not set"
  exit 1
fi

# Synchronize models from S3 bucket to the /tmp directory
aws s3 sync $MODEL_REPOSITORY /tmp/model_repository

# Start the Triton Inference Server with the model repository located in /tmp and explicitly load the sdxl_scribble_controlnet model
/opt/tritonserver/bin/tritonserver --model-repository=/tmp/model_repository --model-control-mode=explicit --load-model=sdxl_scribble_controlnet
