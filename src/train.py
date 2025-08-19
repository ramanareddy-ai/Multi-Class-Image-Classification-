"""
Training Module for Multi-Class Image Classification
Handles model training with proper logging and monitoring
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path

from config import Config
from model import CNNModel, create_callbacks
from data_preprocessing import prepare_data_for_training
from utils import set_random_seed, create_directories, plot_training_history

class ModelTrainer:
    """Class to handle model training and evaluation"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.model = None
        self.history = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self.callbacks = None

        # Set up directories
        self.config.create_directories()

        # Set random seed for reproducibility
        set_random_seed(self.config.RANDOM_SEED)

        # Configure GPU
        self._configure_gpu()

    def _configure_gpu(self):
        """Configure GPU settings"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s)")

                # Enable mixed precision if specified
                if self.config.MIXED_PRECISION:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    print("Mixed precision training enabled")

            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPUs found, using CPU")

    def prepare_data(self):
        """Prepare data generators for training"""
        print("Preparing data...")
        train_gen, val_gen, test_gen, preprocessor = prepare_data_for_training(self.config)

        self.train_generator = train_gen
        self.validation_generator = val_gen
        self.test_generator = test_gen

        # Update number of classes based on data
        if train_gen:
            self.config.NUM_CLASSES = train_gen.num_classes
            print(f"Updated number of classes to: {self.config.NUM_CLASSES}")

        return preprocessor

    def create_model(self, model_type='custom'):
        """Create and compile the model"""
        print(f"Creating {model_type} model...")
        model_builder = CNNModel(self.config)
        self.model = model_builder.create_model(model_type)

        print("Model created successfully!")
        print(f"Total parameters: {self.model.count_params():,}")

        # Create callbacks
        self.callbacks = create_callbacks(self.config)

        return self.model

    def train_model(self, epochs=None, verbose=1):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        if self.train_generator is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        if epochs is None:
            epochs = self.config.EPOCHS

        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")

        start_time = time.time()

        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=self.callbacks,
            verbose= 1,
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time/3600:.2f} hours")

        # Save final model
        model_path = self.config.get_model_path(f"{self.config.MODEL_NAME}_final.h5")
        self.model.save(model_path)
        print(f"Final model saved to: {model_path}")

        # Save training history
        self._save_training_history()

        return self.history

    def _save_training_history(self):
        """Save training history to JSON file"""
        if self.history is None:
            return

        history_path = self.config.LOGS_DIR / 'training_history.json'

        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"Training history saved to: {history_path}")

    def evaluate_model(self, test_generator=None):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not created or trained.")

        if test_generator is None:
            test_generator = self.test_generator

        if test_generator is None:
            print("No test data available for evaluation.")
            return None

        print("Evaluating model on test data...")

        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(
            test_generator,
            verbose=1
        )

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        # Generate predictions for detailed analysis
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)

        # Get true classes
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())

        # Generate classification report
        from sklearn.metrics import classification_report, confusion_matrix

        print("\nDetailed Classification Report:")
        report = classification_report(
            true_classes, predicted_classes,
            target_names=class_labels,
            digits=4
        )
        print(report)

        # Save evaluation results
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }

        eval_path = self.config.LOGS_DIR / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        return evaluation_results

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return

        plot_training_history(self.history, save_path=self.config.LOGS_DIR)

    def fine_tune_model(self, learning_rate=1e-5, epochs=10):
        """Fine-tune the model with lower learning rate"""
        if self.model is None:
            raise ValueError("Model not created or trained.")

        print(f"Starting fine-tuning with learning rate: {learning_rate}")

        # Unfreeze some layers if using transfer learning
        if hasattr(self.model.layers[0], 'trainable'):
            # Unfreeze top layers
            for layer in self.model.layers[0].layers[-20:]:
                layer.trainable = True

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.config.LOSS_FUNCTION,
            metrics=self.config.METRICS
        )

        # Create new callbacks with different monitor
        fine_tune_callbacks = create_callbacks(self.config)

        # Continue training
        fine_tune_history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=fine_tune_callbacks,
            verbose=1,
            workers=self.config.NUM_WORKERS,
            use_multiprocessing=True
        )

        # Save fine-tuned model
        ft_model_path = self.config.get_model_path(f"{self.config.MODEL_NAME}_fine_tuned.h5")
        self.model.save(ft_model_path)
        print(f"Fine-tuned model saved to: {ft_model_path}")

        return fine_tune_history

def main():
    """Main training function"""
    print("Starting Multi-Class Image Classification Training")
    print("=" * 60)

    # Initialize configuration
    config = Config()

    # Initialize trainer
    trainer = ModelTrainer(config)

    # Prepare data
    preprocessor = trainer.prepare_data()

    # Create model
    trainer.create_model('custom')  # Options: 'custom', 'resnet', 'efficientnet'

    # Train model
    history = trainer.train_model()

    # Plot training history
    trainer.plot_training_history()

    # Evaluate model
    evaluation_results = trainer.evaluate_model()

    print("\nTraining completed successfully!")
    if evaluation_results:
        print(f"Final Test Accuracy: {evaluation_results['test_accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
