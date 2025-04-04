import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.ensemble import IsolationForest
from utils.data_generator import DefectGenerator
from utils.preprocessing import create_image_pairs
from models.autoencoder import Autoencoder
from models.unet import UNetModel
def generate_synthetic_data(args):
    print(f"Generating synthetic defect data...")
    os.makedirs(args.normal_dir, exist_ok=True)
    os.makedirs(args.synthetic_dir, exist_ok=True)
    generator = DefectGenerator(output_dir=args.synthetic_dir)
    generator.generate_dataset(args.normal_dir, n_per_image=args.n_per_image)
    print(f"Synthetic data generation complete. Check {args.synthetic_dir}")
def train_autoencoder(args):
    print(f"Training autoencoder model")
    normal_images, _, _ = create_image_pairs(
        args.normal_dir, args.synthetic_dir, target_size=(256, 256)
    )
    if not isinstance(normal_images, np.ndarray):
         raise TypeError("create_image_pairs did not return a numpy array for normal_images. Feature extraction needs image data.")

    print(f"Loaded {len(normal_images)} normal images")
    indices = np.arange(len(normal_images))
    np.random.shuffle(indices)
    n_train = int(len(normal_images) * 0.8)
    train_normal = normal_images[indices[:n_train]]
    test_normal = normal_images[indices[n_train:]]
    print(f"AE Training set: {len(train_normal)} images")
    print(f"AE Test set (for threshold): {len(test_normal)} images")

    autoencoder = Autoencoder(input_shape=(256, 256, 3), model_path=args.autoencoder_path)
    autoencoder.build_model()

    history = autoencoder.train(
        train_normal,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    threshold = autoencoder.set_threshold(test_normal, percentile=95)
    print(f"Anomaly threshold set to: {threshold:.6f}")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{args.autoencoder_path}_history.png")
    print(f"Autoencoder training complete. Model saved to {args.autoencoder_path}.h5")
def train_unet(args):
    print(f"Training U-Net model...")
    _, defect_images, masks = create_image_pairs(
        args.normal_dir, args.synthetic_dir, target_size=(256, 256)
    )
    if not isinstance(defect_images, np.ndarray) or not isinstance(masks, np.ndarray):
         raise TypeError("create_image_pairs did not return numpy arrays for defect_images/masks.")

    print(f"Loaded {len(defect_images)} defect images with masks")
    indices = np.arange(len(defect_images))
    np.random.shuffle(indices)
    n_train = int(len(defect_images) * 0.8)
    train_defect = defect_images[indices[:n_train]]
    train_masks = masks[indices[:n_train]]
    test_defect = defect_images[indices[n_train:]]
    test_masks = masks[indices[n_train:]]
    print(f"U-Net Training set: {len(train_defect)} images")
    print(f"U-Net Test set: {len(test_defect)} images")
    unet = UNetModel(input_shape=(256, 256, 3), model_path=args.unet_path)
    unet.build_model()

    history = unet.train(
        train_defect,
        train_masks,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    plt.figure(figsize=(12, 5))
    plt.tight_layout()
    plt.savefig(f"{args.unet_path}_history.png")
    print("Evaluating U-Net on test set...")
    results = unet.evaluate(test_defect, test_masks)
    print(f"U-Net test evaluation:")
    if isinstance(results, dict) and 'evaluation' in results:
        for metric_name, metric_value in results['evaluation'].items():
            print(f"  {metric_name}: {metric_value:.4f}")
    else:
        print("Evaluation did not return expected format.")
    print(f"U-Net training complete. Model saved to {args.unet_path}.h5")
def train_isoforest(args):
    print(f"Training Isolation Forest model...")
    target_size = (256, 256)
    normal_images, _, _ = create_image_pairs(
        args.normal_dir, args.synthetic_dir, target_size=target_size
    )
    if not isinstance(normal_images, np.ndarray):
         raise TypeError("create_image_pairs did not return a numpy array for normal_images. Feature extraction needs image data.")

    print(f"Loaded {len(normal_images)} normal images for Isolation Forest training.")
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg',
                          input_shape=(target_size[0], target_size[1], 3))
    base_model.trainable = False # Freeze weights
    print("ResNet50 feature extractor loaded.")
    print("Extracting features from normal images...")
    batch_size = args.batch_size
    all_features = []
    num_batches = int(np.ceil(len(normal_images) / batch_size))

    for i in tqdm(range(num_batches)):
        batch_indices = range(i * batch_size, min((i + 1) * batch_size, len(normal_images)))
        img_batch_0_255 = (normal_images[batch_indices] * 255.0).astype(np.float32)
        img_batch_preprocessed = resnet_preprocess(img_batch_0_255)
        features = base_model.predict(img_batch_preprocessed, verbose=0)
        all_features.append(features)
    all_features = np.vstack(all_features)
    print(f"Extracted features shape: {all_features.shape}")
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(all_features)
    print("Isolation Forest training complete.")
    model_dir = os.path.dirname(args.isoforest_path)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(iso_forest, args.isoforest_path)
    print(f"Isolation Forest model saved to {args.isoforest_path}")
def main():
    parser = argparse.ArgumentParser(description='Train defect detection models')
    parser.add_argument('--normal_dir', type=str, default='data/raw',
                        help='Directory containing normal (defect-free) images')
    parser.add_argument('--synthetic_dir', type=str, default='data/synthetic',
                        help='Directory to store generated synthetic defect images')
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate synthetic defect data before training')
    parser.add_argument('--n_per_image', type=int, default=5,
                        help='Number of synthetic variations to generate per normal image')
    parser.add_argument('--train_autoencoder', action='store_true',
                        help='Train the autoencoder model')
    parser.add_argument('--train_unet', action='store_true',
                        help='Train the U-Net model')
    parser.add_argument('--train_isoforest', action='store_true',
                        help='Train the ResNet50+Isolation Forest model')
    parser.add_argument('--autoencoder_path', type=str, default='models/saved/autoencoder',
                        help='Path prefix to save/load autoencoder model (.h5)')
    parser.add_argument('--unet_path', type=str, default='models/saved/unet',
                        help='Path prefix to save/load U-Net model (.h5)')
    parser.add_argument('--isoforest_path', type=str, default='models/saved/isoforest.joblib',
                        help='Path to save/load Isolation Forest model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (for AE and U-Net)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.autoencoder_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.unet_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.isoforest_path), exist_ok=True)
    if args.generate_data:
        generate_synthetic_data(args)
    if args.train_autoencoder:
        train_autoencoder(args)
    if args.train_unet:
        train_unet(args)
    if args.train_isoforest:
        train_isoforest(args)
    if not (args.generate_data or args.train_autoencoder or args.train_unet or args.train_isoforest):
        print("No actions specified. Use --generate_data, --train_autoencoder, --train_unet, or --train_isoforest")
        parser.print_help()
if __name__ == "__main__":
    main()