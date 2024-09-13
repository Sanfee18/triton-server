FROM nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3

# Set the working directory where the repository will be cloned
WORKDIR /triton-inference-server

# Clone the repository
RUN git clone https://github.com/Sanfee18/triton-inference-server .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
