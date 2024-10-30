# Setting up Triton Inference Server on EC2

This guide walks you through setting up a Triton Inference Server on an EC2 instance, building the Docker image, and running it with model synchronization from an S3 bucket.

I **strongly recommend** reading the [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/) to gain a solid understanding of each Triton Server component.

> [!Note]
>
> This is **not** a Docker or AWS tutorial. If you’re new to this technologies, I recommend checking out some great resources online to get up to speed before diving in!

---
## Prerequisites 

Before setting up the Triton Inference Server, ensure you meet the following requirements and take time to understand key concepts that influence its operation, as Triton is a complex program:

- **EC2 Instance with GPU support**: You’ll need an instance type that provides at least **16GB of RAM** (such as `g4dn.2xlarge` or `g5.2xlarge`). Instances with 16GB or less may cause the Triton Inference Server to freeze during intensive tasks. Ensure you choose an instance with enough memory and GPU power to run your models smoothly, along with Amazon Linux OS.
- **S3 Bucket**: Your model repository should be stored in an S3 bucket, organized following Triton’s **required structure**. 

    > In the `models` folder of this repository you have the model structure from our project:
    >```bash
    >models
    >  └── sdxl_scribble_controlnet
    >      ├── 1
    >      │   └── model.py
    >      └── config.pbtxt
    >```
    > Please, refer to the [Triton Model Repository Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) for details on model organization.

- **IAM Role with AmazonS3ReadOnlyAccess:** Create or assign an IAM role to your EC2 instance that has the `AmazonS3ReadOnlyAccess` policy attached. This will allow Triton to securely access the model repository stored in your S3 bucket.
- **Understanding Triton Backends**: Triton supports multiple [model backends](https://github.com/triton-inference-server/backend) like PyTorch, TensorFlow, ONNX or Python. It's important to understand these options so you can choose the most appropriate backend for your project, ensuring optimal performance and integration of your models.

---
## Setup Options

You have two options for setting up the Triton Inference Server and FastAPI frontend on your EC2 instance:

1. **[Setup with Docker Compose (Recommended)](#setting-up-with-docker-compose)**: This option simplifies the process, allowing you to configure and run everything with a single command. Docker Compose manages both services, ensuring they start in the correct order. This is ideal if you’re looking to get up and running quickly.

2. **[Manual Setup](#setting-up-manually)**: This option provides a step-by-step guide to setting up each component individually. Following this setup can give you a deeper understanding of the components involved and allows for more granular customization if needed.

Choose the setup that best fits your needs. **We recommend reading through the manual setup at least once** to gain familiarity with the components involved, even if you decide to use Docker Compose.

## `Step 1:` Clone the Repository
---

To begin, clone the project repository from GitHub to your **local machine**. This repository contains all the necessary files for setting up the `Triton Inference Server` and the `FastAPI` frontend.

```bash
git clone https://github.com/Sanfee18/triton-server.git
cd triton-server
```

## `Step 2:` Install Docker on the EC2 instance
---

Launch the EC2 and follow the next steps:

1. **Update the Packages:**

```bash
sudo yum update -y
```

2. **Install Docker:**

```bash
sudo amazon-linux-extras install docker
```

3. **Start the Docker Service:**

```bash
sudo service docker start
```

## `Step 3:` Transfer Files to EC2
---
Next, transfer the necessary files for setting up Triton Inference Server to your EC2 instance. This includes the `run.sh`, `requirements.txt`, `Dockerfile`, and the `docker-compose.yaml` and `triton.env` (skip bouth if you choose manual setup), and FastAPI files (`main.py`, `requirements.txt`, and `Dockerfile`).

1. **Create two Directories on EC2**:

```bash
mkdir /home/ec2-user/triton-server
```

```bash
mkdir /home/ec2-user/triton-server/fastapi-triton
```

2. **Transfer Files Using `scp`**:

On your **local machine**:

```bash
cd /path/to/triton-server
scp -i <your-ec2-key.pem> Dockerfile requirements.txt run.sh docker-compose.yaml triton.env ec2-user@<your-ec2-ip>:/home/ec2-user/triton-server
```

```bash
cd fastapi-triton
scp -i <your-ec2-key.pem> main.py requirements.txt Dockerfile ec2-user@<your-ec2-ip>:/home/ec2-user/triton-server/fastapi-triton/
```

Replace:
- `<your-ec2-key.pem>` with the path to your EC2 key pair file.
- `<your-ec2-ip>` with your EC2 instance’s public IP.

> [!Note]
>
> If you create the directories as the `root` user, you may encounter a `permission denied` error when attempting to transfer files using `scp`. To resolve this, run the following command to change the ownership of the directory to the `ec2-user`:
> ```bash
> sudo chown ec2-user:ec2-user /home/ec2-user/<directory-name>
> ```

---
## `Setting Up with Docker Compose`

As an alternative to manual setup, you can configure and deploy the Triton Inference Server alongside FastAPI using `docker-compose`, which simplifies the setup by running both services with a single command. This section will guide you through modifying the environment file, configuring the `docker-compose` file, and launching your services.

### 1. Modify Environment Variables

In the project’s root directory, you’ll find a file named `triton.env`. To ensure Triton Inference Server loads your models, replace the `MODEL_REPOSITORY` variable in this file with your own S3 bucket URI containing the model repository.

```plaintext
# triton.env
MODEL_REPOSITORY=<s3-bucket-URI>
```

### 2. Health Check and Service Dependencies

To ensure FastAPI only starts after Triton Inference Server is fully operational, the `docker-compose.yaml` file includes a health check. This health check uses an internal Triton endpoint (`http://localhost:8000/v2/health/ready`) that confirms Triton is ready to handle requests before FastAPI starts, avoiding issues with premature API requests.

### Using `docker-compose`

Once you’ve updated `triton.env` with your S3 bucket URI, you can deploy both services by running:

```bash
docker-compose up --build -d
```

This command builds and launches both services in the background. Docker Compose will handle GPU support, environment variables, and the service dependency order, simplifying the setup significantly.

---
## `Setting Up Manually`
### Understanding Dockerfile

> [!Important]
>
> Understanding the steps to create the Triton Docker image is essential for adapting it to your specific needs. Please don’t skip this section.

1. **Choose the correct Base Image:**

Start by choosing a suitable Triton base image from the official [NVIDIA Docker catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags). This base image must align with the backend needed for your model.

Since our models are going to be loaded using `Python` via Hugging Face's diffusers library, we used the Docker image which only supports **PyTorch** and **Python** backends:
```bash
FROM nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3
```

2. **Install AWS CLI:**

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

3. **Add Dependencies:**

Since Triton Docker images don’t come pre-installed with any Python libraries, you’ll need to manually add them to fit your requirements:

```bash
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
```


### `Step 1:` Build the Docker Image
---

Build the Docker image using the transferred files:

```bash
cd /home/ec2-user/triton-server/
sudo docker build -t triton-server:latest -f Dockerfile .
```

> [!Note]
>
> If you're using a Mac with an ARM architecture (M chip), your local Docker build may not be compatible with the x86 architecture of the EC2 instance. That's why we are building the Docker image directly on the EC2 instance to ensure compatibility.

Alternatively, you can build the image on a compatible machine, push it to **Amazon Elastic Container Registry (ECR)**, and then pull it onto your EC2 instance.

### `Step 2:` Run the Docker Container
---

Run the Docker container with GPU support:

```bash
docker run --gpus=all --net=host -d -e MODEL_REPOSITORY=<s3-bucket-URI> triton-server:latest
```
> - Set `--net=host` so FastAPI can access Triton server ports on `localhost`.

Replace:
- `<s3-bucket-URI>` with the URI of your S3 bucket containing the models folder.

#### Verify Server Status

You can verify that the Triton Inference Server has started successfully by checking the Docker logs:
```bash
docker logs -f <docker-container-id>
```

### `Next Step:` Accessing the Triton Inference Server
---

We will be using a [FastAPI frontend](fastapi-triton/) to interact with the Triton Inference Server, making it easier to handle inference requests through a REST API.

#### Open Ports

Triton Inference Server will expose the following ports, which are essential for different types of requests:

- **`8000`**: **HTTP Requests**  
        This port is used for handling HTTP requests, allowing clients to communicate with the server using RESTful APIs for model inference.

- **`8001`**: **gRPC Requests**  
This port enables communication via gRPC, a high-performance RPC framework, ideal for scenarios requiring efficient streaming and low-latency communication.

- **`8002`**: **Metrics**  
This port exposes server metrics, providing valuable insights into the performance and health of the Triton Inference Server, which can be monitored and analyzed.
