"""
from diffusers import DiffusionPipeline
import torch

# Load the base SDXL model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float32,
    height=512,
    width=512,
    num_inference_steps=25,
    )
pipe = pipe.to("cpu")  # or .to("cuda") if using a GPU
pipe.safety_checker = None

prompt = "a group of people camping in a forest, with a nearby lake"
image = pipe(prompt).images[0]

# Save the image
image.save("camping.png")
"""
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32  # or bfloat16 if supported
)

prompt = "Photo of a people camping in the forest with a lake in the foreground and mountains in the background"
image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=60,
    safety_checker=None,
    guidance_scale=8.0,
).images[0]

image.save("output.png")