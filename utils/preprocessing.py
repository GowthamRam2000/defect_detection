import os
import cv2
import numpy as np
from pathlib import Path


def load_and_preprocess(image_path, target_size=(256, 256), grayscale=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size:
        img = cv2.resize(img, target_size)

    img = img.astype(np.float32) / 255.0

    return img


def load_image_mask_pair(image_path, mask_path, target_size=(256, 256)):
    img = load_and_preprocess(image_path, target_size)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
    else:
        mask = cv2.resize(mask, target_size)

    threshold = 128
    mask = (mask > threshold).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return img, mask


def create_image_pairs(normal_dir, defect_dir, mask_dir=None, target_size=(256, 256)):
    normal_images = []
    defect_images = []
    masks = []

    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for normal_file in normal_files:
        try:
            img = load_and_preprocess(normal_file, target_size)
            normal_images.append(img)
        except Exception as e:
            print(f"Error processing {normal_file}: {str(e)}")

    defect_files = [os.path.join(defect_dir, f) for f in os.listdir(defect_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in f]

    for defect_file in defect_files:
        try:
            base_name = Path(defect_file).stem
            if '_defect' in base_name:
                base_name = base_name.replace('_defect', '')

            mask_file = None
            if mask_dir:
                potential_mask = os.path.join(mask_dir, f"{base_name}_mask.png")
                if os.path.exists(potential_mask):
                    mask_file = potential_mask

            if not mask_file:
                potential_mask = os.path.join(defect_dir, f"{base_name}_mask.png")
                if os.path.exists(potential_mask):
                    mask_file = potential_mask

            defect_img = load_and_preprocess(defect_file, target_size)
            defect_images.append(defect_img)

            if mask_file:
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size)
                mask = (mask > 128).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)
            else:
                mask = np.zeros((target_size[0], target_size[1], 1), dtype=np.float32)

            masks.append(mask)

        except Exception as e:
            print(f"Error processing {defect_file}: {str(e)}")

    return np.array(normal_images), np.array(defect_images), np.array(masks)


def prepare_dataset_split(normal_images, defect_images, masks, train_ratio=0.8):
    indices = np.arange(len(normal_images))
    np.random.shuffle(indices)
    normal_images = normal_images[indices]
    n_train = int(len(normal_images) * train_ratio)
    train_normal = normal_images[:n_train]
    test_normal = normal_images[n_train:]
    defect_indices = np.arange(len(defect_images))
    np.random.shuffle(defect_indices)
    defect_images = defect_images[defect_indices]
    masks = masks[defect_indices]
    n_defect_train = int(len(defect_images) * train_ratio)
    train_defect = defect_images[:n_defect_train]
    train_masks = masks[:n_defect_train]
    test_defect = defect_images[n_defect_train:]
    test_masks = masks[n_defect_train:]

    return {
        'train_normal': train_normal,
        'test_normal': test_normal,
        'train_defect': train_defect,
        'train_masks': train_masks,
        'test_defect': test_defect,
        'test_masks': test_masks
    }