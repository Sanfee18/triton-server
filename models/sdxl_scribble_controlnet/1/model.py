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
    buff = BytesIO(base64.b64decode(img))
    image = Image.open(buff)
    return image


def encode_images(images):
    encoded_images = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="WebP")
        img_bytes = base64.b64encode(buffer.getvalue())
        encoded_images.append(img_bytes.decode("utf8"))

    return encoded_images


class TritonPythonModel:

    def initialize(self, args):
        self.controlnet = ControlNetModel.from_pretrained(
            "xinsir/controlnet-scribble-sdxl-1.0",
            torch_dtype=torch.float16,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
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

    def execute(self, requests):
        responses = []
        for request in requests:
            # Extract inputs from the request
            prompt = (
                pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy().item()
            )
            image = (
                pb_utils.get_input_tensor_by_name(request, "image").as_numpy().item()
            )
            conditioning_scale = (
                pb_utils.get_input_tensor_by_name(request, "conditioning_scale")
                .as_numpy()
                .item()
            )

            # Decode base64 string into a PIL image
            image_pil = decode_image(image)

            # Prepare the input arguments for the model
            input_args = {
                "prompt": prompt,
                "negative_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                "image": image_pil,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30,
                "conditioning_scale": conditioning_scale,
            }

            # Call the model
            images = self.pipe(**input_args).images

            encoded_images = encode_images(images)

            response = pb_utils.InferenceResponse(
                [
                    pb_utils.Tensor(
                        "generated_image",
                        np.array(encoded_images).astype(object),
                    )
                ]
            )

            responses.append(response)

        return responses
