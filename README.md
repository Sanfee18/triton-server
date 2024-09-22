# Setting up Triton Inference Server on EC2

This guide walks you through setting up a Triton Inference Server on an EC2 instance, building the Docker image, and running it with model synchronization from an S3 bucket.

---

## Prerequisites

- An EC2 instance with GPU support (such as a `g4dn.xlarge` or `p3` instance type). <--- Create and link documentation for setting up EC2 instance
- A `Dockerfile` and `run.sh` script prepared for building your Triton Inference Server (you can clone these from the repository).
- An S3 bucket containing the model repository structured as expected by the Triton Inference Server. For more details, refer to the [Triton Inference Server Model Repository documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).

  > This repository includes a models folder with an example structure of a model repository. Use this structure as a guide for organizing your own models in the S3 bucket to ensure compatibility with Triton Inference Server.

### Clone the Repository
To get all the **necessary files** for setting up the Triton Inference Server, clone the project repository on your **local machine**: 

```bash
git clone https://github.com/Sanfee18/triton-inference-server.git
cd triton-inference-server
```

---

## Steps

### 1. Update the EC2 instance

Start by updating the EC2 instance to ensure that all packages are up-to-date.

```bash
sudo yum update -y
```

### 2. Install Docker

Next, install Docker using Amazon Linux Extras.

```bash
sudo amazon-linux-extras install docker
```

### 3. Start the Docker Service

After installing Docker, start the Docker service.

```bash
sudo service docker start
```

### 4. Create a Downloads Directory

Create a directory to hold your Dockerfile and other necessary files.

```bash
mkdir downloads
cd downloads/
```

### 5. Transfer Files from Your Local Machine

On your local machine, transfer the `Dockerfile` and `run.sh` to the EC2 instance using `scp`. Replace `<path_to_pem>` with the actual path to your `.pem` file, and `<ec2_public_ip>` with the EC2 instance's public IP address.

```bash
scp -i <path_to_pem_file> Dockerfile run.sh ec2-user@<ec2_public_ip>:/home/ec2-user/downloads
```

### 6. Change Directory to Downloads on the EC2 Instance

After transferring the files, navigate to the `downloads` directory on the EC2 instance.

```bash
cd /home/ec2-user/downloads/
```

### 7. Build the Docker Image

Build the Docker image using the `Dockerfile` that was transferred. This image includes the Triton Inference Server and all the necessary dependencies.

```bash
sudo docker build -t ec2-triton:latest -f Dockerfile .
```

> Because you may be using a Mac with an ARM architecture (M chip), the Docker image you build locally may not be compatible with the x86 architecture of the EC2 instance. Therefore, it is recommended to build the Docker image directly on the EC2 instance where it will be deployed.

Alternatively, you could build the image on a compatible machine, upload it to Amazon Elastic Container Registry (ECR), and then pull it from there onto the EC2 instance. This guide shows the manual process of building the image directly on the EC2 instance for simplicity.

### 8. Run the Docker Container

Run the Docker container with GPU support and expose the Triton Inference Server on the appropriate ports (8000 for HTTP, 8001 for gRPC, 8002 for metrics). 

> Note that you have to set `--net=host` so FastAPI can access this ports from the localhost.

```bash
docker run --gpus=all -e MODEL_REPOSITORY=s3://<s3-bucket-name>/models \
--net=host -p 8000:8000 -p 8001:8001 -p 8002:8002 ec2-triton:latest
```
> You have to specify the `MODEL_REPOSITORY` variable for the `run.sh` script to be able to load the models from your S3 bucket.

### 9. Accessing the Triton Inference Server

Once the server is running, you can access:
- **Inference requests**: http://<ec2_public_ip>:8000/v2/models
- **gRPC requests**: Port 8001 (gRPC clients required).
- **Metrics**: http://<ec2_public_ip>:8002/metrics

Ensure that your security group allows inbound traffic on these ports.
