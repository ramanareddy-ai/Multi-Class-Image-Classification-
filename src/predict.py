"""
Prediction Module for Multi-Class Image Classification
Provides inference capabilities for trained models
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import json

from config import Config
from utils import load_and_preprocess_single_image

class ImageClassifier:
    """Class for making predictions on new images"""

    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.model = None
        self.class_names = self.config.CLASS_NAMES

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model for inference"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from: {model_path}")
            print(f"Model input shape: {self.model.input_shape}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict_single_image(self, image_path, top_k=3):
        """Predict class for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Preprocess image
        processed_image = load_and_preprocess_single_image(
            image_path, self.config.IMAGE_SIZE
        )

        if processed_image is None:
            return None

        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)

        # Get top-k predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        top_probabilities = predictions[0][top_indices]

        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
            results.append({
                'rank': i + 1,
                'class_name': class_name,
                'class_index': int(idx),
                'probability': float(prob),
                'confidence_percentage': float(prob * 100)
            })

        return results

    def predict_batch(self, image_paths, batch_size=32):
        """Predict classes for multiple images"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        all_results = []

        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_paths = []

            # Load and preprocess batch
            for path in batch_paths:
                processed_image = load_and_preprocess_single_image(
                    path, self.config.IMAGE_SIZE
                )
                if processed_image is not None:
                    batch_images.append(processed_image[0])  # Remove batch dimension
                    valid_paths.append(path)

            if not batch_images:
                continue

            # Convert to numpy array
            batch_images = np.array(batch_images)

            # Make predictions
            predictions = self.model.predict(batch_images, verbose=0)

            # Process results
            for j, (path, pred) in enumerate(zip(valid_paths, predictions)):
                top_idx = np.argmax(pred)
                class_name = self.class_names[top_idx] if top_idx < len(self.class_names) else f"Class_{top_idx}"

                all_results.append({
                    'image_path': str(path),
                    'predicted_class': class_name,
                    'class_index': int(top_idx),
                    'confidence': float(pred[top_idx]),
                    'confidence_percentage': float(pred[top_idx] * 100),
                    'all_probabilities': pred.tolist()
                })

        return all_results

    def visualize_prediction(self, image_path, show_top_k=3):
        """Visualize prediction results"""
        # Get prediction
        results = self.predict_single_image(image_path, top_k=show_top_k)

        if results is None:
            print(f"Could not process image: {image_path}")
            return

        # Load and display image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(image_path)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Display image
        ax1.imshow(image)
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Display predictions
        classes = [r['class_name'] for r in results]
        probabilities = [r['confidence_percentage'] for r in results]
        colors = ['green' if i == 0 else 'orange' if i == 1 else 'red' for i in range(len(classes))]

        bars = ax2.barh(classes, probabilities, color=colors, alpha=0.7)
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_title(f'Top {show_top_k} Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)

        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax2.text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

        # Print results
        print(f"\nPrediction Results for: {Path(image_path).name}")
        print("-" * 50)
        for result in results:
            print(f"Rank {result['rank']}: {result['class_name']} "
                  f"({result['confidence_percentage']:.2f}%)")

    def predict_from_webcam(self, duration=10):
        """Predict from webcam feed (requires webcam)"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print(f"Starting webcam prediction for {duration} seconds...")
        print("Press 'q' to quit early")

        import time
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make prediction
            results = self.predict_single_image(rgb_frame, top_k=1)

            if results:
                pred_class = results[0]['class_name']
                confidence = results[0]['confidence_percentage']

                # Draw prediction on frame
                text = f"{pred_class}: {confidence:.1f}%"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display frame
            cv2.imshow('Webcam Classification', frame)

            # Break conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if time.time() - start_time > duration:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam prediction completed")

    def save_predictions_to_file(self, predictions, output_path):
        """Save predictions to JSON file"""
        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)

        print(f"Predictions saved to: {output_path}")

    def create_prediction_report(self, image_paths, output_dir=None):
        """Create a comprehensive prediction report"""
        if output_dir is None:
            output_dir = self.config.BASE_DIR / 'prediction_results'

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Make predictions
        print("Making predictions...")
        predictions = self.predict_batch(image_paths)

        # Create summary statistics
        class_counts = {}
        for pred in predictions:
            class_name = pred['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Create report
        report = {
            'summary': {
                'total_images': len(predictions),
                'successful_predictions': len([p for p in predictions if p['confidence'] > 0.5]),
                'average_confidence': np.mean([p['confidence'] for p in predictions]),
                'class_distribution': class_counts
            },
            'predictions': predictions,
            'model_info': {
                'model_input_shape': self.model.input_shape if self.model else None,
                'number_of_classes': len(self.class_names),
                'class_names': self.class_names
            },
            'timestamp': str(np.datetime64('now'))
        }

        # Save report
        report_path = output_dir / 'prediction_report.json'
        self.save_predictions_to_file(report, report_path)

        # Create visualization
        self._visualize_prediction_distribution(class_counts, output_dir)

        print(f"Prediction report created in: {output_dir}")
        return report

    def _visualize_prediction_distribution(self, class_counts, output_dir):
        """Visualize prediction distribution"""
        plt.figure(figsize=(12, 8))

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.bar(classes, counts)
        plt.title('Prediction Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')

        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main prediction function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Image Classification Prediction')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--image', help='Path to single image for prediction')
    parser.add_argument('--batch', help='Path to directory containing images')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for prediction')
    parser.add_argument('--output', help='Output directory for results')

    args = parser.parse_args()

    # Initialize classifier
    config = Config()
    classifier = ImageClassifier(args.model, config)

    if args.image:
        # Single image prediction
        classifier.visualize_prediction(args.image)

    elif args.batch:
        # Batch prediction
        image_paths = []
        batch_dir = Path(args.batch)

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(batch_dir.glob(ext))
            image_paths.extend(batch_dir.glob(ext.upper()))

        if image_paths:
            classifier.create_prediction_report(image_paths, args.output)
        else:
            print(f"No images found in {args.batch}")

    elif args.webcam:
        # Webcam prediction
        classifier.predict_from_webcam()

    else:
        print("Please specify --image, --batch, or --webcam option")

if __name__ == "__main__":
    main()
