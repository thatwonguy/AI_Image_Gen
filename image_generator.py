# from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import torch
import streamlit as st

@st.cache_resource
def pipe_model():
    model_id = "playgroundai/playground-v2-1024px-aesthetic"
    pipe = DiffusionPipeline.from_pretrained(model_id,
                                            torch_dtype=torch.float32,
                                            use_safetensors=True,
                                            add_watermarker=False,
                                            variant="fp16"
                                            )
    return pipe

prompt = st.text_input("Write a description of the image you want to generate:")

pipe = pipe_model()

if prompt:
    image = pipe(prompt).images[0]
    st.image(image, use_column_width=True)
