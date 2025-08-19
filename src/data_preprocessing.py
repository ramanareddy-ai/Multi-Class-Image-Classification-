"""
Data Preprocessing Module for Multi-Class Image Classification
Includes data loading, augmentation, and preprocessing pipeline
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from utils import set_random_seed


class DataPreprocessor:
    """Data preprocessing class for image classification"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

    def _validation_dir_has_images(self) -> bool:
        """Return True if VALIDATION_DIR contains class subfolders with files."""
        vdir = self.config.VALIDATION_DIR
        if not vdir.exists():
            return False
        for cls_dir in vdir.iterdir():
            if cls_dir.is_dir():
                files = list(cls_dir.glob("*.[jp][pn]g")) + list(cls_dir.glob("*.jpeg"))
                if len(files) > 0:
                    return True
        return False

    def create_data_generators(self):
        """Create data generators with augmentation (robust + safe)."""
        # --- Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            shear_range=self.config.SHEAR_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            horizontal_flip=self.config.HORIZONTAL_FLIP,
            fill_mode=self.config.FILL_MODE,
            validation_split=self.config.VALIDATION_SPLIT,  # only used if fallback needed
        )

        # --- Validation/Test data generator (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # --- TRAIN: from TRAIN_DIR using subset='training'
        self.train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_DIR,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=self.config.RANDOM_SEED,
        )

        # Guard: must have at least 1 sample
        if getattr(self.train_generator, "samples", 0) == 0:
            raise RuntimeError(
                "Training dataset is empty. Put images under data/train/<class>/..."
            )

        # Lock class mapping across all generators
        train_classes = sorted(self.train_generator.class_indices.keys())

        # --- VALIDATION
        if self._validation_dir_has_images():
            self.validation_generator = test_datagen.flow_from_directory(
                self.config.VALIDATION_DIR,
                target_size=self.config.IMAGE_SIZE,
                batch_size=self.config.BATCH_SIZE,
                class_mode="categororical" if False else "categorical",  # keep categorical
                classes=train_classes,   # lock mapping
                shuffle=False,
            )
        else:
            self.validation_generator = train_datagen.flow_from_directory(
                self.config.TRAIN_DIR,
                target_size=self.config.IMAGE_SIZE,
                batch_size=self.config.BATCH_SIZE,
                class_mode="categorical",
                subset="validation",
                shuffle=False,
                seed=self.config.RANDOM_SEED,
            )

        # Safety: if validation is still empty, raise a helpful error
        if getattr(self.validation_generator, "samples", 0) == 0:
            raise RuntimeError(
                "Validation dataset is empty. "
                "Either put at least 1 image per class under data/validation/ "
                "or increase VALIDATION_SPLIT and keep enough images in data/train/."
            )

        # --- TEST (optional)
        self.test_generator = None
        if self.config.TEST_DIR.exists():
            # Check for any test images under known class names
            any_test = False
            for cls in train_classes:
                cls_dir = self.config.TEST_DIR / cls
                if cls_dir.is_dir():
                    count = len(list(cls_dir.glob("*.[jp][pn]g"))) + len(list(cls_dir.glob("*.jpeg")))
                    if count > 0:
                        any_test = True
                        break
            if any_test:
                self.test_generator = test_datagen.flow_from_directory(
                    self.config.TEST_DIR,
                    target_size=self.config.IMAGE_SIZE,
                    batch_size=self.config.BATCH_SIZE,
                    class_mode="categorical",
                    classes=train_classes,  # lock mapping
                    shuffle=False,
                )

        return self.train_generator, self.validation_generator, self.test_generator

    def create_advanced_augmentation(self):
        """Create advanced augmentation pipeline using Albumentations"""
        transform = A.Compose(
            [
                A.Resize(*self.config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2
                ),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.PiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.3,
                ),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform

    def load_and_preprocess_image(self, image_path, target_size=None):
        """Load and preprocess a single image"""
        if target_size is None:
            target_size = self.config.IMAGE_SIZE

        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize (cv2 expects (width, height); IMAGE_SIZE is (h, w), but square is fine)
        image = cv2.resize(image, target_size)

        # Normalize
        image = image.astype(np.float32) / 255.0

        return image

    def create_dataset_from_directory(self, data_dir, validation_split=0.2):
        """Create dataset from directory structure (numpy pipeline alternative)"""
        images = []
        labels = []
        class_names = []

        # Get class directories
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        class_names = [d.name for d in class_dirs]

        print(f"Found {len(class_names)} classes: {class_names}")

        # Load images and labels
        for class_idx, class_dir in enumerate(class_dirs):
            image_files = list(class_dir.glob("*.[jp][pn]g")) + list(class_dir.glob("*.jpeg"))
            print(f"Loading {len(image_files)} images from {class_dir.name}")
            for image_file in tqdm(image_files, desc=f"Loading {class_dir.name}"):
                try:
                    image = self.load_and_preprocess_image(image_file)
                    images.append(image)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Convert labels to categorical
        labels_categorical = to_categorical(labels, num_classes=len(class_names))

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images,
            labels_categorical,
            test_size=validation_split,
            random_state=self.config.RANDOM_SEED,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )

        return (X_train, y_train), (X_val, y_val), class_names

    def analyze_dataset(self, data_generator):
        """Analyze dataset statistics"""
        print("Dataset Analysis:")
        print(f"Number of classes: {data_generator.num_classes}")
        print(f"Class names: {list(data_generator.class_indices.keys())}")
        print(f"Total samples: {data_generator.samples}")
        print(f"Batch size: {data_generator.batch_size}")
        print(f"Image shape: {data_generator.image_shape}")

        # Class distribution
        class_counts = {}
        for class_name, class_idx in data_generator.class_indices.items():
            class_dir = Path(data_generator.directory) / class_name
            count = len(list(class_dir.glob("*.[jp][pn]g")) + list(class_dir.glob("*.jpeg")))
            class_counts[class_name] = count

        print("\nClass distribution:")
        total = sum(class_counts.values()) or 1
        for class_name, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"{class_name}: {count} images ({percentage:.1f}%)")

        return class_counts

    def create_sample_batch_visualization(self, generator, num_images=9):
        """Visualize a sample batch from the generator"""
        batch_images, batch_labels = next(generator)

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()

        class_names = list(generator.class_indices.keys())

        for i in range(min(num_images, len(batch_images))):
            axes[i].imshow(batch_images[i])
            class_idx = np.argmax(batch_labels[i])
            class_name = class_names[class_idx]
            axes[i].set_title(f"Class: {class_name}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(self.config.BASE_DIR / "sample_batch.png", dpi=150, bbox_inches="tight")
        plt.show()


def prepare_data_for_training(config=None):
    """Main function to prepare data for training"""
    if config is None:
        config = Config()

    # Set random seed
    set_random_seed(config.RANDOM_SEED)

    # Create directories
    config.create_directories()

    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)

    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen, test_gen = preprocessor.create_data_generators()

    # Analyze datasets
    if train_gen:
        print("\n" + "=" * 50)
        print("TRAINING DATA")
        print("=" * 50)
        preprocessor.analyze_dataset(train_gen)

    if val_gen:
        print("\n" + "=" * 50)
        print("VALIDATION DATA")
        print("=" * 50)
        preprocessor.analyze_dataset(val_gen)

    return train_gen, val_gen, test_gen, preprocessor


if __name__ == "__main__":
    config = Config()
    train_gen, val_gen, test_gen, preprocessor = prepare_data_for_training(config)

    if train_gen:
        print("\nCreating sample batch visualization...")
        preprocessor.create_sample_batch_visualization(train_gen)
