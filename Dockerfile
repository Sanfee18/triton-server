FROM nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3

# Set the working directory
WORKDIR /opt/tritonserver/backends/python

# Clone the repository
RUN git clone https://github.com/Sanfee18/triton-inference-server

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/tritonserver/backends/python/triton-inference-server/requirements.txt
