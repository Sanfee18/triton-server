# Setting up Triton Inference Server on EC2

This guide walks you through setting up a Triton Inference Server on an EC2 instance, building the Docker image, and running it with model synchronization from an S3 bucket.

> [!Note]
>
> This is **not** a Docker or AWS tutorial. If you’re new to Docker or AWS, I recommend checking out some great resources online to get up to speed before diving in!

---
## Prerequisites 

<!-- TODO: Add section on how to configure the EC2 instance -->

- An EC2 instance with GPU support (such as a `g4dn.xlarge` or `p3` instance type). 
- An S3 bucket containing the model repository structured as expected by the Triton Inference Server. For more details, refer to the [Triton Inference Server Model Repository documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).

  > This repository includes a models folder with an example structure of a model repository. Use this structure as a guide for organizing your own models in the S3 bucket to ensure compatibility with Triton Inference Server.
  >```bash
  >models
  >  └── sdxl_scribble_controlnet
  >      ├── 1
  >      │   └── model.py
  >      └── config.pbtxt
  >```

---
## Clone the repository 

On your **local machine**, clone the project repository with all the **necessary files** for setting up the `Triton Inference Server`: 

```bash.
git clone https://github.com/Sanfee18/triton-inference-server.git
cd triton-inference-server
```

---
## Understanding Dockerfile

It's really important that you understand the steps involved in creating the Triton Docker image.

**1. Choose a Base Image**

First, you have to understand the different [backends](https://github.com/triton-inference-server/backend) supported by Triton.

Then, you should be able to determine which [official NVIDIA Docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags) suits your needs the best.

Since our models are going to be loaded using Python via Hugging Face's diffusers library, we used the Docker image which only supports PyTorch and Python backends:
```bash
FROM nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3
```

**2. Install AWS CLI**

We then install the AWS CLI commands on the Docker: 

```bash
RUN apt-get update && apt-get install -y \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -f awscliv2.zip
```

This will allow the `run.sh` file to access AWS commands to sync the S3 bucket model repository inside the `/tmp/model-repository` folder everytime the container is run:

```bash
# run.sh
aws s3 sync $MODEL_REPOSITORY /tmp/model_repository
```

**3. Add Dependencies**

Triton Docker image doesn't include any preinstalled Python libraries, so you'll need to add the necessary ones for your model or business logic.

```bash
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
```

## `Step 1:` Install Docker on the EC2 instance
---

Launch the EC2 and follow the next steps:

**1. Update the Packages**

```bash
sudo yum update -y
```

**2. Install Docker**

```bash
sudo amazon-linux-extras install docker
```

**3. Start the Docker Service**

```bash
sudo service docker start
```

## `Step 2:` Clone the necessary files to EC2
---

If you've cloned this repository, you may have the necessary files to build the Docker image. Then follow this steps:

**1. Create a Downloads Directory on the EC2**

```bash
mkdir /home/ec2-user/downloads
cd /home/ec2-user/downloads/
```

**2. Transfer Files from Your Local Machine**

Make sure to be inside the `triton-inference-server` folder on your `local machine` and **execute** this command:
```bash
scp -i <path_to_pem_file> Dockerfile requirements.txt run.sh ec2-user@<ec2_public_ip>:/home/ec2-user/downloads
```
> Replace `<path_to_pem>` with the actual path to your `.pem` file, and `<ec2_public_ip>` with the EC2 instance's public IP address.

## `Step 3:` Build the Docker Image
---

Build the Docker image using the `Dockerfile` that was transferred. This image includes the Triton Inference Server and all the necessary dependencies specified on the `requirements.txt` file.

```bash
sudo docker build -t ec2-triton:latest -f Dockerfile .
```

> Because you may be using a Mac with an ARM architecture (M chip), the Docker image you build locally may not be compatible with the x86 architecture of the EC2 instance. Therefore, it is recommended to build the Docker image directly on the EC2 instance where it will be deployed.

Alternatively, you could build the image on a compatible machine, upload it to `Amazon Elastic Container Registry (ECR)`, and then pull it from there onto the EC2 instance. This guide shows the manual process of building the image directly on the EC2 instance for simplicity.

## `Step 4:` Run the Docker Container
---

Run the Docker container with GPU support and expose the Triton Inference Server on the appropriate ports (8000 for HTTP, 8001 for gRPC, 8002 for metrics). 

> Note that you have to set `--net=host` so FastAPI can access this ports from the localhost.

```bash
docker run --gpus=all -e MODEL_REPOSITORY=s3://<s3-bucket-name>/models \
--net=host -p 8000:8000 -p 8001:8001 -p 8002:8002 ec2-triton:latest
```
> You have to specify the `MODEL_REPOSITORY` environment variable for the `run.sh` script to be able to load the models from your S3 bucket.

---
## Accessing the Triton Inference Server

You may want use a [client library](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/README.html) to perform the requests to the Triton Inference Server.

Once the server is running, you can access:
- **Inference requests**: http://<ec2_public_ip>:8000/v2/models
- **gRPC requests**: Port 8001 (gRPC clients required).
- **Metrics**: http://<ec2_public_ip>:8002/metrics

Ensure that your security group allows inbound traffic on these ports.
