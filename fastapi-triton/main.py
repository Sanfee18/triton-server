from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tritonclient.http as httpclient
import numpy as np

app = FastAPI()

TRITON_URL = "localhost:8000"
MODEL_NAME = "sdxl_scribble_controlnet"


class InferenceRequest(BaseModel):
    prompt: str
    image: str  # Base64 encoded image
    conditioning_scale: float


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
            httpclient.InferInput("prompt", [1, 1], "BYTES"),
            httpclient.InferInput("image", [1, 1], "BYTES"),
            httpclient.InferInput("conditioning_scale", [1, 1], "FP32"),
        ]

        # Prepare output tensors
        outputs = [
            httpclient.InferRequestedOutput(name="generated_image", binary_data=False)
        ]

        # Set data for the input tensors
        prompt = np.array([[request.prompt]], dtype=np.object_)
        image = np.array([[request.image]], dtype=np.object_)
        conditioning_scale = np.array([[request.conditioning_scale]], dtype=np.float32)

        inputs[0].set_data_from_numpy(prompt)
        inputs[1].set_data_from_numpy(image)
        inputs[2].set_data_from_numpy(conditioning_scale)

        # Send inference request to Triton server
        response = triton_client.infer(MODEL_NAME, inputs=inputs, outputs=outputs)

        # Process the response
        generated_image_str = response.get_output("generated_image")

        # Return the result
        return {"generated_image": generated_image_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI application with the command:
# uvicorn main:app --host 0.0.0.0 --port 8080
