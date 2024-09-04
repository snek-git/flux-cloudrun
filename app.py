import os
import logging
import sys
import time

import streamlit as st
import torch
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Constants
MODEL_PATH = "/app/models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.info(f"Starting application. Device: {DEVICE}")

@st.cache_resource
def load_models():
    logging.info("Loading models...")
    t5 = load_t5(DEVICE, max_length=256)
    logging.info("T5 model loaded")
    clip = load_clip(DEVICE)
    logging.info("CLIP model loaded")
    model = load_flow_model("flux-schnell", device=DEVICE)
    logging.info("Flow model loaded")
    ae = load_ae("flux-schnell", device=DEVICE)
    logging.info("Autoencoder loaded")
    logging.info("All models loaded successfully")
    return model, ae, t5, clip

def generate_image(prompt, width, height, num_steps, guidance, seed):
    logging.info(f"Generating image. Prompt: {prompt}, Size: {width}x{height}, Steps: {num_steps}, Guidance: {guidance}, Seed: {seed}")
    model, ae, t5, clip = load_models()

    with torch.inference_mode():
        x = get_noise(1, height, width, device=DEVICE, dtype=torch.bfloat16, seed=seed)
        timesteps = get_schedule(num_steps, (x.shape[-1] * x.shape[-2]) // 4, shift=False)
        inp = prepare(t5=t5, clip=clip, img=x, prompt=prompt)
        x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            x = ae.decode(x)

    x = x.clamp(-1, 1)
    x = x[0].permute(1, 2, 0)
    img = Image.fromarray(((x + 1) * 127.5).cpu().byte().numpy())
    logging.info("Image generated successfully")
    return img

def main():
    st.title("Flux Image Generation on Cloud Run")

    logging.info("Streamlit app started")

    prompt = st.text_input("Enter a prompt", "a photo of a forest with mist swirling around the tree trunks")
    width = st.slider("Width", min_value=128, max_value=1024, value=512, step=64)
    height = st.slider("Height", min_value=128, max_value=1024, value=512, step=64)
    num_steps = st.slider("Number of steps", min_value=1, max_value=50, value=20)
    guidance = st.slider("Guidance", min_value=1.0, max_value=10.0, value=3.5)
    seed = st.number_input("Seed", value=42, step=1)

    if st.button("Generate Image"):
        logging.info("Generate Image button clicked")
        with st.spinner("Generating image..."):
            img = generate_image(prompt, width, height, num_steps, guidance, seed)

        st.image(img, caption=prompt)
        logging.info("Image displayed in Streamlit")

if __name__ == "__main__":
    main()