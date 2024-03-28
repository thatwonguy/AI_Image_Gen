from diffusers import StableDiffusionPipeline
import torch
import streamlit as st

@st.cache_resource
def pipe_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    # pipe = pipe.to("cuda")
    return pipe

prompt = st.text_input("Write a description of the image you want to generate:")

pipe = pipe_model()

if prompt:
    image = pipe(prompt).images[0]
    st.image(image, use_column_width=True)
