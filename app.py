import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt

from models.autoencoder import Autoencoder
from models.unet import UNetModel
from utils.preprocessing import load_and_preprocess

st.set_page_config(
    page_title="Visual Product Defect Detection for CV project ",
    page_icon="🔍",
    layout="wide"
)


def load_models():
    models = {}
    autoencoder = Autoencoder()
    try:
        autoencoder.build_model()
        autoencoder.load_weights()
        autoencoder.set_threshold(np.zeros((1, 256, 256, 3)), percentile=95)
        models['autoencoder'] = autoencoder
    except Exception as e:
        st.warning(f"Failed to load autoencoder model: {str(e)}")

    unet = UNetModel()
    try:
        unet.build_model()
        unet.load_weights()
        models['unet'] = unet
    except Exception as e:
        st.warning(f"Failed to load UNet model: {str(e)}")

    return models


def process_image_autoencoder(image, model):
    processed_img = np.expand_dims(image, axis=0)
    results = model.detect_anomalies(processed_img)
    is_anomaly = results['is_anomaly'][0]
    anomaly_score = results['anomaly_score'][0]
    reconstruction = results['reconstruction'][0]
    error_image = results['error_images'][0]
    anomaly_mask = results['anomaly_masks'][0]
    error_image_normalized = (error_image - error_image.min()) / (error_image.max() - error_image.min() + 1e-8)
    heatmap = cv2.applyColorMap((error_image_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    alpha = 0.7
    blended = cv2.addWeighted(
        (image * 255).astype(np.uint8),
        1 - alpha,
        heatmap,
        alpha,
        0
    )
    return {
        'is_anomaly': is_anomaly,
        'anomaly_score': anomaly_score,
        'reconstruction': reconstruction,
        'error_image': error_image_normalized,
        'anomaly_mask': anomaly_mask,
        'heatmap': heatmap / 255.0,
        'blended': blended / 255.0
    }

def process_image_unet(image, model):
    processed_img = np.expand_dims(image, axis=0)
    pred_masks, binary_masks = model.predict(processed_img)
    pred_mask = pred_masks[0].squeeze()
    binary_mask = binary_masks[0].squeeze()
    heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    alpha = 0.7
    blended = cv2.addWeighted(
        (image * 255).astype(np.uint8),
        1 - alpha,
        heatmap,
        alpha,
        0
    )
    is_anomaly = binary_mask.sum() > 0
    return {
        'is_anomaly': is_anomaly,
        'pred_mask': pred_mask,
        'binary_mask': binary_mask,
        'heatmap': heatmap / 255.0,
        'blended': blended / 255.0
    }


def main():
    st.title("Visual Product Defect Detection")
    if 'models' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.models = load_models()

    models = st.session_state.models
    if not models:
        st.error("No models loaded. Please check model paths and try again.")
        return
    st.sidebar.title("Settings")

    model_type = st.sidebar.radio(
        "Select Detection Method",
        ["Autoencoder (Anomaly Detection)", "U-Net (Segmentation)"]
    )
    uploaded_file = st.sidebar.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
    if model_type == "Autoencoder (Anomaly Detection)":
        threshold_percentile = st.sidebar.slider(
            "Anomaly Threshold Percentile",
            min_value=80,
            max_value=99,
            value=95,
            step=1
        )
        if 'autoencoder' in models:
            models['autoencoder'].threshold = np.percentile(
                [0.01, 0.05],
                threshold_percentile
            )
    else:
        segmentation_threshold = st.sidebar.slider(
            "Segmentation Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            image = np.array(pil_image) / 255.0
            image = cv2.resize(image, (256, 256))
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with st.spinner("Processing image..."):
                if model_type == "Autoencoder (Anomaly Detection)" and 'autoencoder' in models:
                    results = process_image_autoencoder(image, models['autoencoder'])

                    with col2:
                        st.subheader("Results")
                        if results['is_anomaly']:
                            st.error(" Defect Detected!")
                        else:
                            st.success(" No Defect Detected")

                        st.write(f"Anomaly Score: {results['anomaly_score']:.6f}")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("Reconstruction")
                        st.image(results['reconstruction'], use_column_width=True)

                    with col2:
                        st.subheader("Error Heatmap")
                        st.image(results['heatmap'], use_column_width=True)

                    with col3:
                        st.subheader("Overlay")
                        st.image(results['blended'], use_column_width=True)

                elif model_type == "U-Net (Segmentation)" and 'unet' in models:
                    results = process_image_unet(image, models['unet'])

                    with col2:
                        st.subheader("Results")
                        if results['is_anomaly']:
                            st.error("⚠️ Defect Detected!")
                        else:
                            st.success("✅ No Defect Detected")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("Defect Probability Map")
                        st.image(results['pred_mask'], use_column_width=True)

                    with col2:
                        st.subheader("Defect Heatmap")
                        st.image(results['heatmap'], use_column_width=True)

                    with col3:
                        st.subheader("Overlay")
                        st.image(results['blended'], use_column_width=True)

                else:
                    st.error(f"Selected model '{model_type}' not available")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    with st.expander("How it works"):
        st.write("""
        ### Visual Product Defect Detection System

        This system uses two different approaches to detect defects in product images:

        1. **Autoencoder (Anomaly Detection)**: 
           - Trained only on normal (defect-free) images
           - Learns to reconstruct normal patterns 
           - Defects appear as reconstruction errors
           - Thresholding determines if an image contains defects

        2. **U-Net (Segmentation)**:
           - Trained on pairs of defective images and their defect masks
           - Directly predicts defect regions in the image
           - More precise localization of defects

        #### Usage:
        1. Upload a product image using the sidebar
        2. Select the detection method
        3. Adjust the threshold as needed
        4. View results showing defect detection and visualization
        """)


if __name__ == "__main__":
    main()