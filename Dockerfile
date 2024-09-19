FROM nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3

# INSTALL AWS CLI
RUN apt-get update && apt-get install -y \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -f awscliv2.zip

# INSTALL DEPENDENCIES
RUN pip install torch torchvision torchaudio
RUN pip install transformers diffusers
RUN pip install xformers accelerate

# COPY ENTRYPOINT FILE
COPY run.sh /tmp/run.sh
RUN chmod +x /tmp/run.sh

# ENTRYPOINT TO RUN THE SCRIPT
ENTRYPOINT ["/tmp/run.sh"]
