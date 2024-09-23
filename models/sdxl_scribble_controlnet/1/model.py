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
            prompt = (
                pb_utils.get_input_tensor_by_name(request, "prompt")
                .as_numpy()
                .item()
                .decode("utf-8")
            )
            image = (
                pb_utils.get_input_tensor_by_name(request, "image")
                .as_numpy()
                .item()
                .decode("utf-8")
            )
            conditioning_scale = pb_utils.get_input_tensor_by_name(
                request, "conditioning_scale"
            )

            if not prompt or not image:
                # If there is an error, there is no need to pass the
                # "output_tensors" to the InferenceResponse. The "output_tensors"
                # that are passed in this case will be ignored.
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "The prompt or image is empty or not provided."
                        )
                    )
                )
                continue

            # create the arguments for the generation
            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

            input_args = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=1024,
                height=1024,
                num_inference_steps=30,
            )

            if conditioning_scale:
                input_args["conditioning_scale"] = conditioning_scale

            image = decode_image(image)

            images = self.pipe(**input_args).images

            encoded_images = encode_images(images)

            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor(
                            "generated_image", np.array(encoded_images).astype(object)
                        )
                    ]
                )
            )

        return responses
