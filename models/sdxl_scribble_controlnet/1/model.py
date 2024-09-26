import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers import EulerAncestralDiscreteScheduler

from io import BytesIO
import base64
from PIL import Image


def decode_image(img):
    buff = BytesIO(base64.b64decode(img.encode("utf8")))
    image = Image.open(buff)
    return image


def encode_images(images):
    encoded_images = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue())
        encoded_images.append(img_str.decode("utf8"))

    return encoded_images


class TritonPythonModel:

    def initialize(self, args):
        controlnet = ControlNetModel.from_pretrained(
            "xinsir/controlnet-scribble-sdxl-1.0",
            torch_dtype=torch.float16,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        # change vae of sdxl
        self.pipe.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )

        # change scheduler
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
        )

        # send pipe to gpu
        self.pipe.to("cuda")
        # enable xformers (optional), requires xformers installation
        # https://github.com/facebookresearch/xformers#installing-xformers
        self.pipe.unet.enable_xformers_memory_efficient_attention()
        # cpu offload for memory saving, requires accelerate>=0.17.0
        self.pipe.enable_model_cpu_offload()

    def execute(self, requests):
        responses = []
        for request in requests:
            # Debug: Print the incoming request details
            print("Incoming request:")
            print(request)

            try:
                # Extract inputs from the request
                prompt = (
                    pb_utils.get_input_tensor_by_name(request, "prompt")
                    .as_numpy()
                    .item()
                    .decode("utf-8")
                )
                image_str = (
                    pb_utils.get_input_tensor_by_name(request, "image")
                    .as_numpy()
                    .item()
                    .decode("utf-8")
                )
                conditioning_scale = (
                    pb_utils.get_input_tensor_by_name(request, "conditioning_scale")
                    .as_numpy()
                    .item()
                )

                # Debug: Print extracted inputs
                print(f"Prompt: {prompt}")
                print(
                    f"Image string: {image_str[:30]}..."
                )  # Print only the first 30 characters
                print(f"Conditioning scale: {conditioning_scale}")

                # Decode the Base64 image string to a PIL image
                image = decode_image(image_str)

                # Convert the PIL image to a NumPy array
                image_np = np.array(image)

                # Add a batch dimension to the image shape
                image_np = np.expand_dims(
                    image_np, axis=0
                )  # Shape should now be [1, height, width, channels]

                # Prepare the input arguments for the model
                input_args = {
                    "prompt": prompt,
                    "negative_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                    "image": image_np,
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 30,
                    "conditioning_scale": conditioning_scale,
                }

                # Debug: Print input arguments before processing
                print("Input arguments for model:", input_args)

                # Call the model
                images = self.pipe(**input_args).images

                encoded_images = encode_images(images)

                responses.append(
                    pb_utils.InferenceResponse(
                        [
                            pb_utils.Tensor(
                                "generated_image",
                                np.array(encoded_images).astype(object),
                            )
                        ]
                    )
                )
            except Exception as e:
                print(f"Error processing request: {e}")
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"Failed to process request: {e}")
                    )
                )

        return responses
