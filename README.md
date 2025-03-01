# Visual Product Defect Detection System
A deep learning system for detecting manufacturing defects in product images using two complementary approaches: an autoencoder-based anomaly detector and a U-Net segmentation model.
## Overview
This project implements a visual inspection system that can identify defects in manufactured products. It uses two distinct deep learning approaches:
1. **Autoencoder-based Anomaly Detection**: Trained exclusively on defect-free (normal) images, this model learns to reconstruct normal patterns. When presented with defective items, the reconstruction error highlights anomalous regions.
2. **U-Net Segmentation**: This segmentation approach directly predicts defect masks, providing precise localization of defects. It's trained on pairs of defective images and their corresponding defect masks.
The web interface allows users to upload product images and receive instant visual feedback on detected defects.
## Dataset
This project uses the MVTec Anomaly Detection (MVTec AD) dataset, which is specifically designed for anomaly detection in industrial inspection scenarios.
> **MVTec AD Dataset Citation:**  
> Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:  
> The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;  
> in: International Journal of Computer Vision, January 2021  
> Available at: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
The MVTec AD dataset contains 15 different object categories with over 5,000 high-resolution images. Each category contains normal (defect-free) training images and test images with various types of defects along with pixel-precise ground truth annotations.
## Features
- **Dual Detection Methods**: Toggle between anomaly detection and segmentation approaches
- **Interactive Web Interface**: Upload images and visualize results in real-time
- **Defect Heatmaps**: Visual representation of detected anomalies
- **Adjustable Thresholds**: Fine-tune detection sensitivity
- **Synthetic Data Generation**: Create realistic defects for training
## Installation
```bash
# Clone the repository
git clone https://github.com/GowthamRam2000/defect_detection.git
cd defect_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## Usage
### Data Preparation
1. Place defect-free product images in `data/raw/`
2. Generate synthetic defects:
   ```bash
   python train.py --generate_data
   ```
### Training Models
```bash
# Train both models
python train.py --train_autoencoder --train_unet
# Or train individually
python train.py --train_autoencoder
python train.py --train_unet
```
### Running the Application
```bash
streamlit run app.py
```
## Model Architecture
### Autoencoder
- Convolutional encoder-decoder architecture
- Trained only on normal (defect-free) images
- Defects appear as high reconstruction errors
- Image-level and pixel-level anomaly detection
### U-Net
- Standard U-Net architecture with skip connections
- Trained on defective images with corresponding masks
- Direct prediction of defect regions
- Precise segmentation of anomalous areas

## Acknowledgments

- This project builds on the [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- Inspired by research in unsupervised anomaly detection for industrial inspection

## License

This project is available under the MIT License - see the LICENSE file for details.

The MVTec AD dataset used in this project is provided by MVTec and is subject to its own license terms, available in the dataset documentation.
