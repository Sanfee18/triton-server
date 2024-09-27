# Setting up Triton Inference Server on EC2

This guide walks you through setting up a Triton Inference Server on an EC2 instance, building the Docker image, and running it with model synchronization from an S3 bucket.

I **strongly recommend** reading the [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/) to gain a solid understanding of each Triton Server component.
> [!Note]
>
> This is **not** a Docker or AWS tutorial. If you’re new to this technologies, I recommend checking out some great resources online to get up to speed before diving in!

---
## Prerequisites 

Before setting up the Triton Inference Server, ensure you meet the following requirements and understand key concepts that influence its operation:

- **EC2 Instance with GPU support**: You’ll need an instance type such as `g4dn.xlarge` or `p3` that provides sufficient GPU resources to run Triton, along with Amazon Linux OS.

- **S3 Bucket**: Your model repository should be stored in an S3 bucket, organized following Triton’s required structure. Refer to the [Triton Model Repository Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) for details on model organization.

  > This repository contains our example model structure:
  >```bash
  >models
  >  └── sdxl_scribble_controlnet
  >      ├── 1
  >      │   └── model.py
  >      └── config.pbtxt
  >```
  > Use this structure as a template to organize your models for compatibility with Triton.

- **Backends in Triton Inference Server**: Triton supports multiple [model backends](https://github.com/triton-inference-server/backend) (such as PyTorch, TensorFlow, ONNX or Python).
> You'll be selecting the appropriate backend later in this guide when configuring the Dockerfile.

---
## Clone the Repository

To begin, clone the project repository from GitHub to your **local machine**. This repository contains all the necessary files for setting up the `Triton Inference Server` and the `FastAPI` frontend.

### Steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sanfee18/triton-inference-server.git
   ```

2. **Navigate into the project directory**:
   ```bash
   cd triton-inference-server
   ```

Ensure you review the repository structure to understand the components before proceeding with the setup.

---
## Understanding Dockerfile

It's really important that you understand the steps involved in creating the Triton Docker image.

**1. Choose a Base Image**

Start by choosing a suitable Triton base image from the official [NVIDIA Docker catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags). This base image must align with the backend needed for your model.

Since our models are going to be loaded using `Python` via Hugging Face's diffusers library, we used the Docker image which only supports **PyTorch** and **Python** backends:
```bash
FROM nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3
```

**2. Install AWS CLI**

To synchronize models from an S3 bucket, install AWS CLI inside the Docker container:

```bash
RUN apt-get update && apt-get install -y \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -f awscliv2.zip
```

This setup ensures that the `run.sh` script, executed when the container starts, can pull models from your S3 bucket into the local `/tmp/model_repository` directory:

```bash
# run.sh
aws s3 sync $MODEL_REPOSITORY /tmp/model_repository
```

**3. Add Dependencies**

Since Triton Docker images don’t come pre-installed with any Python libraries, you’ll need to manually add them to fit your requirements:

```bash
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
```

## `Step 1:` Install Docker on the EC2 instance
---

Launch the EC2 and follow the next steps:

**1. Update the Packages:**

```bash
sudo yum update -y
```

**2. Install Docker:**

```bash
sudo amazon-linux-extras install docker
```

**3. Start the Docker Service:**

```bash
sudo service docker start
```

## `Step 2:` Transfer the Necessary Files to EC2
---

Once the EC2 instance is set up, transfer the essential files (`run.sh`, `requirements.txt`, and `Dockerfile`) for building the Docker image.

### Steps:

**1. Create a triton-server Directory on the EC2:**

```bash
mkdir /home/ec2-user/triton-server
cd /home/ec2-user/triton-server/
```

**2. Transfer Files using `scp`:**

```bash
scp -i <your-ec2-key.pem> Dockerfile requirements.txt run.sh ec2-user@<your-ec2-ip>:/home/ec2-user/triton-server
```
Replace:
- `<your-ec2-key.pem>` with the path to your EC2 key pair file.
- `<your-ec2-ip>` with the public IP of your EC2 instance.

## `Step 3:` Build the Docker Image
---

Build the Docker image using the transferred `Dockerfile`:

```bash
sudo docker build -t triton-server:latest -f Dockerfile .
```
> [!Note]
>
> If you're using a Mac with an ARM architecture (M chip), your local Docker build may not be compatible with the x86 architecture of the EC2 instance. That's why we are building the Docker image directly on the EC2 instance to ensure compatibility.

Alternatively, you can build the image on a compatible machine, push it to **Amazon Elastic Container Registry (ECR)**, and then pull it onto your EC2 instance.

## `Step 4:` Run the Docker Container
---

Run the Docker container with GPU support:

```bash
docker run --gpus=all ---net=host -d -e MODEL_REPOSITORY=s3://<s3-bucket-name>/models triton-server:latest
```
Replace:
- `<s3-bucket-name>` with the name of your S3 bucket containing the models folder.

> [!Note]
>
> - You have to set `--net=host` so FastAPI can access Triton server ports on `localhost`.
> - You also need to specify the `MODEL_REPOSITORY` environment variable for the `run.sh` script to be able to load the models from your S3 bucket.

### Verify Server Status

You can verify that the Triton Inference Server has started successfully by checking the Docker logs:
```bash
docker logs -f <docker-container-id>
```
> You can retrieve the `docker-container-id` using `docker ps` command.

---
## `Last Step:` Accessing the Triton Inference Server

We will be using a [FastAPI frontend](fastapi-triton/) to interact with the Triton Inference Server, making it easier to handle inference requests through a REST API.

### Open Ports

Triton Inference Server will expose the following ports:
- `8000`: http requests
- `8001`: gRPC requests
- `8002`: Metrics
