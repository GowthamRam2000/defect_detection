import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt
import joblib 

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from models.autoencoder import Autoencoder
from models.unet import UNetModel

st.set_page_config(
    page_title="Visual Product Defect Detection",
    page_icon="🔍",
    layout="wide"
)

def load_models():
    models={}
    target_size=(256, 256) 

    autoencoder_path='models/saved/autoencoder.h5'
    autoencoder=Autoencoder(input_shape=(target_size[0], target_size[1], 3))
    try:
        if os.path.exists(autoencoder_path):
            autoencoder.build_model()
            autoencoder.load_weights(autoencoder_path)

            dummy_normal=np.random.rand(2, target_size[0], target_size[1], 3) * 0.1
            try:
                autoencoder.set_threshold(dummy_normal, percentile=95)
            except Exception as thresh_e:
                 st.warning(f"Could not set initial AE threshold: {thresh_e}. Relying on slider.")
            models['autoencoder']=autoencoder
            st.success("Autoencoder model loaded.")
        else:
             st.warning(f"Autoencoder model file not found at {autoencoder_path}")
    except Exception as e:
        st.error(f"Failed to load autoencoder model: {str(e)}")
    unet_path='models/saved/unet.h5'
    unet=UNetModel(input_shape=(target_size[0], target_size[1], 3))
    try:
        if os.path.exists(unet_path):
            unet.build_model()
            unet.load_weights(unet_path)
            models['unet']=unet
            st.success("U-Net model loaded.")
        else:
             st.warning(f"U-Net model file not found at {unet_path}")
    except Exception as e:
        st.error(f"Failed to load UNet model: {str(e)}")
    isoforest_path='models/saved/isoforest.joblib'
    try:
        if os.path.exists(isoforest_path):
            iso_forest_model=joblib.load(isoforest_path)
            feature_extractor=ResNet50(weights='imagenet', include_top=False, pooling='avg',
                                          input_shape=(target_size[0], target_size[1], 3))
            feature_extractor.trainable=False
            models['isoforest']={
                'model': iso_forest_model,
                'extractor': feature_extractor
            }
            st.success("Isolation Forest model and ResNet50 extractor loaded.")
        else:
             st.warning(f"Isolation Forest model file not found at {isoforest_path}")
    except Exception as e:
        st.error(f"Failed to load Isolation Forest model/extractor: {str(e)}")


    return models

def process_image_autoencoder(image, model):
    processed_img=np.expand_dims(image, axis=0)
    results=model.detect_anomalies(processed_img, pixel_threshold=None)

    is_anomaly=results['is_anomaly'][0]
    anomaly_score=results['anomaly_score'][0]
    reconstruction=results['reconstruction'][0]
    error_image=results['error_images'][0] 
    anomaly_mask=results['anomaly_masks'][0]

    error_image_normalized=(error_image - np.min(error_image)) / (np.max(error_image) - np.min(error_image) + 1e-8)
    heatmap=cv2.applyColorMap((error_image_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap=cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    alpha=0.6 
    blended=cv2.addWeighted(
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
    processed_img=np.expand_dims(image, axis=0)
    pred_masks, binary_masks=model.predict(processed_img, threshold=0.5) 

    pred_mask=pred_masks[0].squeeze()
    binary_mask=binary_masks[0].squeeze() 

    heatmap=cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap=cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    alpha=0.6
    blended=cv2.addWeighted(
        (image * 255).astype(np.uint8),
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    is_anomaly=np.sum(binary_mask) > 0 

    return {
        'is_anomaly': is_anomaly,
        'pred_mask': pred_mask,
        'binary_mask': binary_mask, 
        'heatmap': heatmap / 255.0, 
        'blended': blended / 255.0
    }

def process_image_isoforest(image, models_dict):
    if 'isoforest' not in models_dict:
        st.error("Isolation Forest model not loaded.")
        return None

    iso_forest_model=models_dict['isoforest']['model']
    feature_extractor=models_dict['isoforest']['extractor']
    target_size=feature_extractor.input_shape[1:3]
    if image.shape[:2] != target_size:
         img_resized=cv2.resize(image, target_size)
    else:
         img_resized=image.copy()

    img_0_255=(img_resized * 255.0).astype(np.float32)
    img_preprocessed=resnet_preprocess(img_0_255)
    img_batch=np.expand_dims(img_preprocessed, axis=0)
    features=feature_extractor.predict(img_batch, verbose=0)

    anomaly_score=iso_forest_model.decision_function(features)[0]
    prediction=iso_forest_model.predict(features)[0]

    is_anomaly=prediction == -1 
    return {
        'is_anomaly': is_anomaly,
        'anomaly_score': anomaly_score,
    }

def main():
    st.title("Visual Product Defect Detection")

    if 'models' not in st.session_state:
        with st.spinner("Loading models... This might take a moment."):
            st.session_state.models=load_models()

    models=st.session_state.models
    if not models:
        st.error("No models could be loaded. Please check model paths/files and restart.")
        st.stop() 
    st.sidebar.title("Settings")
    model_type=st.sidebar.radio(
        "Select Detection Method",
        ["Autoencoder (Anomaly Detection)",
         "U-Net (Segmentation)",
         "ResNet50 + Isolation Forest"]
    )

    uploaded_file=st.sidebar.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
    ae_threshold_percentile=95
    unet_segmentation_threshold=0.5

    if model_type == "Autoencoder (Anomaly Detection)":

        ae_raw_threshold=st.sidebar.slider(
            "Anomaly Score Threshold (Higher=More Anomalous)",
            min_value=float(models.get('autoencoder').threshold * 0.5) if 'autoencoder' in models else 0.0,
            max_value=float(models.get('autoencoder').threshold * 5.0) if 'autoencoder' in models else 0.1,
            value=float(models.get('autoencoder').threshold) if 'autoencoder' in models else 0.01,
            step=0.001,
            format="%.4f"
        )
        if 'autoencoder' in models:
            models['autoencoder'].threshold=ae_raw_threshold

    elif model_type == "U-Net (Segmentation)":
        unet_segmentation_threshold=st.sidebar.slider(
            "Segmentation Threshold (Probability > Threshold=Defect)",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )

    if uploaded_file is not None:
        try:
            image_bytes=uploaded_file.read()
            pil_image=Image.open(io.BytesIO(image_bytes)).convert('RGB') 
            image_np=np.array(pil_image)
            target_size=(256, 256)
            image_resized=cv2.resize(image_np, target_size)
            image_normalized=image_resized.astype(np.float32) / 255.0
            st.subheader("Original Image")
            st.image(image_normalized, use_column_width=True, caption=f"Original ({image_np.shape[1]}x{image_np.shape[0]}) -> Resized ({target_size[1]}x{target_size[0]})")


            with st.spinner("Processing image..."):
                results=None
                if model_type == "Autoencoder (Anomaly Detection)":
                    if 'autoencoder' in models:
                        results=process_image_autoencoder(image_normalized, models['autoencoder'])
                    else:
                        st.error("Autoencoder model not available.")

                    if results:
                        st.subheader("Autoencoder Results")
                        col1, col2=st.columns([1, 2]) 
                        with col1:
                             if results['is_anomaly']:
                                 st.error(" Defect Detected!")
                             else:
                                 st.success(" No Defect Detected")
                             st.write(f"Anomaly Score: {results['anomaly_score']:.6f}")
                             st.write(f"(Threshold: {models['autoencoder'].threshold:.6f})")

                        with col2:
                             st.image(results['blended'], caption="Original + Error Heatmap Overlay", use_column_width=True)
                        with st.expander("Show More Details (Autoencoder)"):
                             col_exp1, col_exp2, col_exp3=st.columns(3)
                             with col_exp1:
                                 st.image(results['reconstruction'], caption="Reconstruction", use_column_width=True)
                             with col_exp2:
                                 st.image(results['error_image'], caption="Normalized Error Map", use_column_width=True)
                             with col_exp3:
                                st.image(results['heatmap'], caption="Error Heatmap", use_column_width=True)


                elif model_type == "U-Net (Segmentation)":
                    if 'unet' in models:


                        def process_image_unet_with_threshold(image, model, threshold):
                            processed_img=np.expand_dims(image, axis=0)
                            pred_masks, binary_masks=model.predict(processed_img, threshold=threshold) # Pass threshold
                            pred_mask=pred_masks[0].squeeze()
                            binary_mask=binary_masks[0].squeeze()
                            heatmap=cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            heatmap=cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                            alpha=0.6
                            blended=cv2.addWeighted((image * 255).astype(np.uint8), 1 - alpha, heatmap, alpha, 0)
                            is_anomaly=np.sum(binary_mask) > 0
                            return {
                                'is_anomaly': is_anomaly, 'pred_mask': pred_mask, 'binary_mask': binary_mask,
                                'heatmap': heatmap / 255.0, 'blended': blended / 255.0
                            }

                        results=process_image_unet_with_threshold(image_normalized, models['unet'], unet_segmentation_threshold)
                    else:
                        st.error("U-Net model not available.")

                    if results:
                        st.subheader("U-Net Results")
                        col1, col2=st.columns([1, 2])
                        with col1:
                            if results['is_anomaly']:
                                st.error(" Defect Detected!")
                            else:
                                st.success(" No Defect Detected")
                            st.write(f"(Threshold: {unet_segmentation_threshold:.2f})")

                        with col2:
                            st.image(results['blended'], caption="Original + Defect Heatmap Overlay", use_column_width=True)

                        with st.expander("Show More Details (U-Net)"):
                            col_exp1, col_exp2, col_exp3=st.columns(3)
                            with col_exp1:
                                st.image(results['pred_mask'], caption="Defect Probability Map", use_column_width=True, clamp=True)
                            with col_exp2:
                                st.image(results['binary_mask'], caption="Binary Defect Mask", use_column_width=True)
                            with col_exp3:
                                st.image(results['heatmap'], caption="Defect Heatmap", use_column_width=True)


                elif model_type == "ResNet50 + Isolation Forest":
                    if 'isoforest' in models:
                        results=process_image_isoforest(image_normalized, models)
                    else:
                        st.error("Isolation Forest model not available.")

                    if results:
                        st.subheader("ResNet50 + Isolation Forest Results")
                        if results['is_anomaly']:
                            st.error(" Defect Detected!")
                        else:
                            st.success(" No Defect Detected")
                        st.write(f"Anomaly Score: {results['anomaly_score']:.4f}")
                        st.caption("(Lower score indicates higher likelihood of anomaly)")

                else:
                    st.error(f"Selected model type '{model_type}' processing not implemented.")

        except Exception as e:
            st.error(f"An error occurred processing the image: {str(e)}")
            import traceback
            st.error(traceback.format_exc()) # Show full traceback for debugging

    with st.expander("How it works"):
        st.markdown("""
        ### Visual Product Defect Detection System
Developed as part of CV course at IIT J by Gowtham Ram M Saravanan GS ,Dinesh ,Chella Vignesh 
        This system uses different approaches to detect defects in product images:
        1.  **Autoencoder (Anomaly Detection)**:
            * Trained only on normal (defect-free) images.
            * Learns to reconstruct normal patterns accurately.
            * Defects (anomalies) cause high reconstruction errors.
            * An image-level score (average error) is compared to a threshold.
            * Visualizations show the reconstruction and error heatmap.
            
        2.  **U-Net (Segmentation)**:
            * Trained on pairs of (synthetic) defective images and their pixel-level defect masks.
            * Directly predicts which pixels belong to a defect region (segmentation).
            * Provides precise defect localization.
            * Visualizations show the predicted probability map and heatmap overlay.

        3.  **ResNet50 + Isolation Forest**:
            * Uses a powerful pre-trained network (ResNet50) to extract meaningful features from normal images during training.
            * An Isolation Forest model learns to distinguish these normal features.
            * For a new image, its features are extracted and checked by the Isolation Forest. Features differing significantly from normal are flagged as anomalous.
            * Provides an image-level anomaly score.

        #### Usage:
        1. Select the detection method from the sidebar.
        2. Upload a product image.
        3. Adjust thresholds (if applicable) using the sidebar sliders.
        4. View the detection result and any available visualizations.
        """)


if __name__ == "__main__":
    main()