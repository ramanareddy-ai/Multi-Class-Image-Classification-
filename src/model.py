"""
CNN Model Architecture for Multi-Class Image Classification
Designed to achieve 98%+ accuracy with proper training
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0
import numpy as np

from config import Config

class CNNModel:
    """Custom CNN model for multi-class image classification"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.model = None

    def build_custom_cnn(self):
        """Build a custom CNN architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.config.IMAGE_SIZE, 3)),

            # Data normalization
            layers.Rescaling(1./255),

            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Global Average Pooling (reduces overfitting)
            layers.GlobalAveragePooling2D(),

            # Dense layers with regularization
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            # Output layer
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])

        return model

    def build_resnet_transfer(self):
        """Build model using ResNet50 transfer learning"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMAGE_SIZE, 3)
        )

        # Freeze base model layers
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])

        return model

    def build_efficientnet_transfer(self):
        """Build model using EfficientNetB0 transfer learning"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMAGE_SIZE, 3)
        )

        # Freeze base model layers initially
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])

        return model

    def create_model(self, model_type='custom'):
        """Create and compile the model"""
        if model_type == 'custom':
            self.model = self.build_custom_cnn()
        elif model_type == 'resnet':
            self.model = self.build_resnet_transfer()
        elif model_type == 'efficientnet':
            self.model = self.build_efficientnet_transfer()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss=self.config.LOSS_FUNCTION,
            metrics=self.config.METRICS
        )

        return self.model

    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        return self.model.summary()

    def plot_model_architecture(self, save_path=None):
        """Plot model architecture"""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")

        if save_path is None:
            save_path = self.config.MODELS_DIR / 'model_architecture.png'

        keras.utils.plot_model(
            self.model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB'
        )
        print(f"Model architecture saved to {save_path}")

def create_callbacks(config):
    """Create training callbacks"""
    callbacks = []

    # Model checkpoint
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=config.get_checkpoint_path(),
        monitor=config.MONITOR_METRIC,
        save_best_only=config.SAVE_BEST_ONLY,
        save_weights_only=False,
        mode=config.MODE,
        verbose=1
    )
    callbacks.append(checkpoint)

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.MONITOR_METRIC,
        patience=config.EARLY_STOPPING_PATIENCE,
        mode=config.MODE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Reduce learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=config.MONITOR_METRIC,
        factor=0.2,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        mode=config.MODE,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # TensorBoard logging
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=config.LOGS_DIR,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)

    return callbacks

if __name__ == "__main__":
    # Example usage
    config = Config()
    model_builder = CNNModel(config)

    # Create custom model
    model = model_builder.create_model('custom')
    print("Custom CNN Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")

    # Print model summary
    model_builder.get_model_summary()
