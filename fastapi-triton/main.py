from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tritonclient.http as httpclient
import numpy as np

app = FastAPI()

TRITON_URL = "localhost:8000"
MODEL_NAME = "sdxl_scribble_controlnet"


# Initialize Triton client during app startup
@app.on_event("startup")
async def startup_event():
    global triton_client
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
        if not triton_client.is_server_live():
            raise RuntimeError("Triton server is not live.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Triton client: {str(e)}")


class InferenceRequest(BaseModel):
    prompt: str
    image: str  # Base64 encoded image
    conditioning_scale: float | None = None


@app.get("/health")
async def health_check():
    """
    Check the health of the Triton server and model.
    """
    try:
        if not triton_client.is_server_live():
            raise HTTPException(status_code=503, detail="Triton server is not live.")

        if not triton_client.is_model_ready(MODEL_NAME):
            raise HTTPException(
                status_code=503, detail=f"Model {MODEL_NAME} is not ready."
            )

        return {"status": "Healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sdxl/scribble-controlnet/infer")
async def sdxl_scribble_controlnet(request: InferenceRequest):
    """
    Route for inference using the SDXL ControlNet model.
    """
    try:
        # Prepare input tensors
        inputs = [
            httpclient.InferInput("prompt", [1], "STRING"),
            httpclient.InferInput("image", [1], "STRING"),
        ]

        # Set data for the input tensors
        inputs[0].set_data_from_numpy(np.array([request.prompt], dtype=np.object_))
        inputs[1].set_data_from_numpy(np.array([request.image], dtype=np.object_))

        # Handle optional conditioning scale input
        if request.conditioning_scale is not None:
            conditioning_scale_input = httpclient.InferInput(
                "conditioning_scale", [1], "FP32"
            )
            conditioning_scale_input.set_data_from_numpy(
                np.array([request.conditioning_scale], dtype=np.float32)
            )
            inputs.append(conditioning_scale_input)

        # Send inference request to Triton server
        response = triton_client.infer(MODEL_NAME, inputs=inputs)

        # Process the response
        generated_image = response.as_numpy("generated_image")[0].decode("utf-8")

        return {"generated_image": generated_image}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI application with the command:
# uvicorn main:app --host 0.0.0.0 --port 8000
