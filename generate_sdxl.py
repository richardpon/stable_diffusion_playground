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

prompt = "Realistic Photo of a person doing martial arts, specifically Brazillian Jiu-Jitsu, wearing a white gi, and performing an armbar from guard"
negative_prompt = "low quality, blurry, or bokeh effects"
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_inference_steps=50,
    safety_checker=None,
    guidance_scale=9.0,
).images[0]

image.save("bjj_armbar.png")
