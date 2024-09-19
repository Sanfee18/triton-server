#!/bin/bash

# Verifica que la variable de entorno esté definida
if [ -z "$MODEL_REPOSITORY" ]; then
  echo "MODEL_REPOSITORY is not set"
  exit 1
fi

# Sincroniza los modelos desde S3 al directorio /tmp
aws s3 sync $MODEL_REPOSITORY /tmp/model_repository

# Ejecutar Triton con la carpeta de modelos en /tmp
/opt/tritonserver/bin/tritonserver --model-repository=/tmp/model_repository

