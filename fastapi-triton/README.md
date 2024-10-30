# FastAPI Frontend for Triton Inference Server

This guide will walk you through the process of launching a FastAPI application inside a Docker container on your EC2 instance. This FastAPI app will interact with the Triton Inference Server for running inference on the `sdxl_scribble_controlnet` model.

Make sure to read the [FastAPI Documentation](https://fastapi.tiangolo.com/) for further understanding.

---
## Prerequisites

- Ensure that your [Triton Inference Server is running](..) on the EC2 instance.
- Make sure you've already cloned the repository onto your local machine.

---
## Triton Client Integration on FastAPI

The FastAPI frontend uses the `tritonclient` Python package to interact with the Triton Inference Server, specifically via the [`tritonclient.http`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/_reference/tritonclient/tritonclient.http.html#module-tritonclient.http) module. 

We recommend reading the [Triton client documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/_reference/tritonclient/tritonclient.http.html#module-tritonclient.http) to understand how the Triton client operates, how requests are sent, and how responses are handled. This knowledge will help you in customizing and expanding the FastAPI frontend.

---
## Understanding the FastAPI Dockerfile

It’s important to understand the steps involved in creating the Docker image for running FastAPI, which interacts with the Triton Inference Server.

1. **Choose a Base Image**

    We start by using an official slim Python image as the base. The slim variant is lightweight, ensuring the Docker image remains small while still containing all the necessary tools for a Python environment.

    ```bash
    FROM python:3.10.12-slim
    ```

2. **Set the Working Directory**

    We define `/app` as the working directory where all application files and configurations will reside inside the container. This ensures all subsequent file operations are relative to this directory.

    ```bash
    WORKDIR /app
    ```

3. **Add the FastAPI Application Code**

    After defining the working directory, we copy the FastAPI application code (`main.py`) into the container. This file contains the logic for interacting with Triton via FastAPI.

    ```bash
    COPY main.py .
    ```

4. **Install Dependencies**

    Next, we copy the `requirements.txt` file containing the necessary Python dependencies into the container and install them.

    ```bash
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    ```

5. **Define the Entrypoint**

    Finally, we define the command that will run when the Docker container starts. In this case, the command launches the FastAPI app using `uvicorn`, a lightning-fast ASGI server. We specify that the application will run on host `0.0.0.0` (which makes it accessible externally) and port `8080`.

    ```bash
    ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
    ```

### `Step 1:` Build the FastAPI Docker Image
---

Run the following command inside the `fastapi-triton` directory:

```bash
sudo docker build -t fastapi-triton:latest -f Dockerfile .
```

### `Step 2:` Run the FastAPI Docker Container
---

After the Docker image is built, you can run the FastAPI application on port `8080` using the following command:

```bash
docker run --net=host -d fastapi-triton
```
> The `--net=host` argument ensures that the FastAPI app can communicate with the Triton Inference Server on `localhost`.

### Accessing the FastAPI Endpoint
---

Once the FastAPI container is running, you can access the endpoints that have been set up in the `main.py` file. You can modify or extend these endpoints to suit your specific needs.

- **Inference Request**: This endpoint allows you to send an inference request to the `sdxl_scribble_controlnet` model. The request should include a prompt, a base64-encoded image, and an optional conditioning scale.
```
POST http://<your-ec2-ip>:8080/sdxl/scribble-controlnet/infer
```

The payload format for an inference request is as follows:
```json
{
    "prompt": "A description of the image",
        "image": "base64-encoded image data",
        "conditioning_scale": float number
}
```

### Matching Payloads with Model Configuration

The payload format used in inference requests must align with the configuration defined in the `config.pbtxt` file of your Triton model. This file specifies the number of inputs, their data types, and dimensions. To ensure that your requests are valid, make sure that the input types and shapes in your payloads match those defined in the `config.pbtxt`.

To gain a better understanding of how to configure your models, we recommend reviewing the [Triton Model Configuration Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html). Additionally, for a reference on how the data types specified in Triton's `config.pbtxt` map to `numpy` types used in Python requests, you can consult the [data types mapping table](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html?highlight=dtypes#datatypes).

### Performance Testing with Triton Performance Analyzer

To evaluate the performance of your models, you can use the Triton Inference Server's **Performance Analyzer**. This tool allows you to send test requests to your model and measure various performance metrics. You can run the Performance Analyzer in a Docker container. For detailed instructions on how to use it, refer to the [Performance Analyzer Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/README.html).

## Notes

- The FastAPI container will be running on port `8080`, and the Triton Inference Server should be available on `localhost:8000` inside the container.
