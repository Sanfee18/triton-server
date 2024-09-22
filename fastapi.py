from fastapi import FastAPI
import requests

app = FastAPI()

# Assuming Triton is running on localhost:8000 inside Docker
TRITON_URL = "http://localhost:8000/v2/models"


@app.post("/infer")
async def infer(model_name: str, data: dict):
    triton_endpoint = f"{TRITON_URL}/{model_name}/infer"
    response = requests.post(triton_endpoint, json=data)
    return response.json()
