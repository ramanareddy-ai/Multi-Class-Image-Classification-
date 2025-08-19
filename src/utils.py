"""
Utility functions for the Multi-Class Image Classification project
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
import json
from PIL import Image
import cv2

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def create_directories(dirs):
    """Create directories if they don't exist"""
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def plot_training_history(history, save_path=None):
    """Plot training history with loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(Path(save_path) / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_path:
        plt.savefig(Path(save_path) / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()

def calculate_model_metrics(y_true, y_pred, class_names):
    """Calculate detailed classification metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_metrics': report
    }

    return metrics

def load_and_preprocess_single_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image for prediction"""
    try:
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(image_path)

        # Resize image
        image = cv2.resize(image, target_size)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def visualize_predictions(images, true_labels, predicted_labels, class_names, 
                         num_images=12, save_path=None):
    """Visualize model predictions"""
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.ravel()

    for i in range(min(num_images, len(images))):
        # Display image
        if len(images[i].shape) == 3:
            axes[i].imshow(images[i])
        else:
            axes[i].imshow(images[i].squeeze(), cmap='gray')

        # Get labels
        true_class = class_names[true_labels[i]] if true_labels[i] < len(class_names) else f"Class {true_labels[i]}"
        pred_class = class_names[predicted_labels[i]] if predicted_labels[i] < len(class_names) else f"Class {predicted_labels[i]}"

        # Set title with color coding
        title_color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', 
                         color=title_color, fontsize=10)
        axes[i].axis('off')

    # Hide remaining subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(Path(save_path) / 'prediction_visualization.png', dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")

    plt.show()

def save_model_info(model, history, save_path):
    """Save model information and training history"""
    save_path = Path(save_path)

    # Model summary
    with open(save_path / 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Training history
    if history:
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]

        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)

    # Model parameters
    model_info = {
        'total_parameters': int(model.count_params()),
        'trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        'non_trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])),
        'model_layers': len(model.layers),
    }

    with open(save_path / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Model information saved to: {save_path}")

def create_class_activation_map(model, image, class_idx, layer_name='mixed7'):
    """Create Class Activation Map (CAM) for model interpretability"""
    # Get the model up to the specified layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Get gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]

    # Get gradients of the loss w.r.t. the convolutional layer output
    grads = tape.gradient(loss, conv_outputs)

    # Calculate the mean intensity of the gradients over all feature maps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by the gradients
    conv_outputs = conv_outputs[0]
    conv_outputs *= pooled_grads

    # Create the heatmap
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def get_model_size(model_path):
    """Get model file size in MB"""
    if Path(model_path).exists():
        size_bytes = Path(model_path).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0

def benchmark_model_inference(model, test_images, num_runs=100):
    """Benchmark model inference time"""
    import time

    # Warm-up runs
    for _ in range(10):
        _ = model.predict(test_images[:1], verbose=0)

    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(test_images[:1], verbose=0)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000

    return {
        'average_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'fps': 1000 / avg_time
    }

def print_system_info():
    """Print system information"""
    print("System Information:")
    print("-" * 30)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {os.sys.version}")

    # GPU info
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("No GPUs available")

    # CPU info
    print(f"CPU threads: {os.cpu_count()}")

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    set_random_seed(42)
    print_system_info()
