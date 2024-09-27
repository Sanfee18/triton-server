# Use the official Python image as the base
FROM python:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the FastAPI application code into the container
COPY main.py .

# Install the required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Command to run the FastAPI application
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
