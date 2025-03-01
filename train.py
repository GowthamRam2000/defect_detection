import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.data_generator import DefectGenerator
from utils.preprocessing import create_image_pairs, prepare_dataset_split
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

    print(f"Loaded {len(normal_images)} normal images")

    indices = np.arange(len(normal_images))
    np.random.shuffle(indices)
    n_train = int(len(normal_images) * 0.8)
    train_normal = normal_images[:n_train]
    test_normal = normal_images[n_train:]
    print(f"Training set: {len(train_normal)} images")
    print(f"Test set: {len(test_normal)} images")
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

    print(f"Loaded {len(defect_images)} defect images with masks")
    indices = np.arange(len(defect_images))
    np.random.shuffle(indices)
    n_train = int(len(defect_images) * 0.8)
    train_defect = defect_images[:n_train]
    train_masks = masks[:n_train]
    test_defect = defect_images[n_train:]
    test_masks = masks[n_train:]

    print(f"Training set: {len(train_defect)} images")
    print(f"Test set: {len(test_defect)} images")
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
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('U-Net Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('U-Net Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.unet_path}_history.png")
    results = unet.evaluate(test_defect, test_masks)

    print(f"U-Net test evaluation:")
    for metric_name, metric_value in results['evaluation'].items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print(f"U-Net training complete. Model saved to {args.unet_path}.h5")


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
    parser.add_argument('--autoencoder_path', type=str, default='models/saved/autoencoder',
                        help='Path to save/load autoencoder model')
    parser.add_argument('--unet_path', type=str, default='models/saved/unet',
                        help='Path to save/load U-Net model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.autoencoder_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.unet_path), exist_ok=True)

    if args.generate_data:
        generate_synthetic_data(args)

    if args.train_autoencoder:
        train_autoencoder(args)

    if args.train_unet:
        train_unet(args)

    if not (args.generate_data or args.train_autoencoder or args.train_unet):
        print("No actions specified. Use --generate_data, --train_autoencoder, or --train_unet")
        parser.print_help()


if __name__ == "__main__":
    main()