from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tritonclient.http as httpclient
import numpy as np
import base64

app = FastAPI()

TRITON_URL = "http://localhost:8000"  # Adjust the URL as needed
MODEL_NAME = "sdxl_scribble_controlnet"


class InferenceRequest(BaseModel):
    prompt: str
    image: str  # Base64 encoded image
    conditioning_scale: str = None


@app.get("/health")
async def health_check():
    """
    Check the health of the Triton server and model.
    """
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
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

        inputs[0].set_data_from_numpy(np.array([request.prompt], dtype=np.object))
        inputs[1].set_data_from_numpy(np.array([request.image], dtype=np.object))

        if request.conditioning_scale:
            conditioning_scale_input = httpclient.InferInput(
                "conditioning_scale", [1], "STRING"
            )
            conditioning_scale_input.set_data_from_numpy(
                np.array([request.conditioning_scale], dtype=np.object)
            )
            response = triton_client.infer(
                MODEL_NAME, inputs=[inputs[0], inputs[1], conditioning_scale_input]
            )
        else:
            response = triton_client.infer(MODEL_NAME, inputs=[inputs[0], inputs[1]])

        # Process the response
        generated_image = response.as_numpy("generated_image")[0].decode("utf-8")

        return {"generated_image": generated_image}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI application
# uvicorn main:app --host 0.0.0.0 --port 8000
