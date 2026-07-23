
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from pathlib import Path
from PIL import Image

from torchvision.models import resnet18

from src.model import load_model
from src.utils import predict_image, transform
from src.gradcam import GradCAM, create_overlay

# ==========================================
# Configuration
# ==========================================

st.set_page_config(
    page_title="Explainable Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Explainable Pneumonia Detection from Chest X-Rays")

st.write(
    "Upload a chest X-ray image to classify it as **NORMAL** or **PNEUMONIA** and visualize the model's attention using **Grad-CAM**."
)

# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# Paths
# ==========================================

PROJECT_DIR = Path(__file__).parent

MODEL_PATH = PROJECT_DIR / "models" / "best_model.pth"

# ==========================================
# Load Model
# ==========================================

model = load_model(
    MODEL_PATH,
    device
)

grad_cam = GradCAM(
    model=model,
    target_layer=model.layer4
)

# ==========================================
# Upload Image
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    prediction, confidence = predict_image(
        model,
        image,
        device
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    heatmap = grad_cam.generate(input_tensor)

    colored_heatmap, overlay = create_overlay(
        image.resize((224,224)),
        heatmap
    )

    st.success(
        f"Prediction: {prediction}"
    )

    st.info(
        f"Confidence: {confidence*100:.2f}%"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            colored_heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    with col3:
        st.image(
            overlay,
            caption="Overlay",
            use_container_width=True
        )
