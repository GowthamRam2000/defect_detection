import os
import cv2
import numpy as np
import random
from pathlib import Path
import albumentations as A


class DefectGenerator:
    def __init__(self, output_dir='data/synthetic'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
        ])

    def add_scratch(self, image, mask=None, count=None):
        if mask is None:
            mask = np.zeros_like(image)[:, :, 0]

        height, width = image.shape[:2]
        if count is None:
            count = random.randint(1, 3)

        for _ in range(count):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)

            color = random.randint(180, 255)
            thickness = random.randint(1, 3)

            cv2.line(image, (x1, y1), (x2, y2), (color, color, color), thickness)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness + 2)

        return image, mask

    def add_spot(self, image, mask=None, count=None):
        if mask is None:
            mask = np.zeros_like(image)[:, :, 0]

        height, width = image.shape[:2]
        if count is None:
            count = random.randint(1, 5)

        for _ in range(count):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(5, 20)
            color = random.randint(180, 255)

            cv2.circle(image, (x, y), radius, (color, color, color), -1)
            cv2.circle(mask, (x, y), radius + 2, 255, -1)

        return image, mask

    def add_crack(self, image, mask=None, count=None):
        if mask is None:
            mask = np.zeros_like(image)[:, :, 0]

        height, width = image.shape[:2]
        if count is None:
            count = random.randint(1, 2)

        for _ in range(count):
            x = random.randint(width // 4, 3 * width // 4)
            y = random.randint(height // 4, 3 * height // 4)
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(30, 100)
            branches = random.randint(2, 5)

            points = [(x, y)]
            for _ in range(branches):
                new_angle = angle + random.uniform(-np.pi / 4, np.pi / 4)
                end_x = int(x + length * np.cos(new_angle))
                end_y = int(y + length * np.sin(new_angle))
                points.append((end_x, end_y))

            for i, (px, py) in enumerate(points):
                if i == 0:
                    continue
                thickness = random.randint(1, 3)
                color = random.randint(180, 255)
                cv2.line(image, (x, y), (px, py), (color, color, color), thickness)
                cv2.line(mask, (x, y), (px, py), 255, thickness + 2)

        return image, mask

    def generate_defect_image(self, input_path, defect_types=None, save=True):
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image at {input_path}")
        defect_image = image.copy()
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if defect_types is None:
            defect_types = random.sample(['scratch', 'spot', 'crack'],
                                         k=random.randint(1, 3))

        for defect_type in defect_types:
            if defect_type == 'scratch':
                defect_image, mask = self.add_scratch(defect_image, mask)
            elif defect_type == 'spot':
                defect_image, mask = self.add_spot(defect_image, mask)
            elif defect_type == 'crack':
                defect_image, mask = self.add_crack(defect_image, mask)
        transformed = self.transform(image=defect_image, mask=mask)
        defect_image = transformed['image']
        mask = transformed['mask']

        if save:
            input_name = Path(input_path).name
            base_name = Path(input_path).stem
            defect_path = os.path.join(self.output_dir, f"{base_name}_defect.jpg")
            cv2.imwrite(defect_path, defect_image)
            mask_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask)

            return defect_path, mask_path

        return defect_image, mask

    def generate_dataset(self, input_dir, n_per_image=5):
        input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        generated_pairs = []

        for input_file in input_files:
            for i in range(n_per_image):
                defect_types = random.sample(['scratch', 'spot', 'crack'],
                                             k=random.randint(1, 3))
                try:
                    defect_path, mask_path = self.generate_defect_image(
                        input_file, defect_types)
                    generated_pairs.append((defect_path, mask_path))
                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")

        return generated_pairs


if __name__ == "__main__":
    generator = DefectGenerator()
    generator.generate_dataset("data/raw", n_per_image=3)