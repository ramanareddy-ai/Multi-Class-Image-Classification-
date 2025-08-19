"""
Model Evaluation Module for Multi-Class Image Classification
Provides comprehensive model evaluation and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime

from config import Config
from utils import plot_confusion_matrix, visualize_predictions, calculate_model_metrics

class ModelEvaluator:
    """Class for comprehensive model evaluation"""

    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.model = None
        self.model_path = model_path

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def evaluate_on_generator(self, test_generator, save_results=True):
        """Evaluate model on test data generator"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print("Evaluating model on test data...")

        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)

        # Get true classes
        true_classes = test_generator.classes
        class_names = list(test_generator.class_indices.keys())

        # Calculate basic metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision = precision_score(true_classes, predicted_classes, average='weighted')
        recall = recall_score(true_classes, predicted_classes, average='weighted')
        f1 = f1_score(true_classes, predicted_classes, average='weighted')

        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Detailed classification report
        report = classification_report(
            true_classes, predicted_classes, 
            target_names=class_names, 
            digits=4
        )
        print(f"\nClassification Report:\n{report}")

        # Plot confusion matrix
        self.plot_confusion_matrix(true_classes, predicted_classes, class_names)

        # Calculate per-class metrics
        per_class_metrics = self.calculate_per_class_metrics(
            true_classes, predicted_classes, class_names
        )

        # Save results if requested
        if save_results:
            self.save_evaluation_results({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': report,
                'per_class_metrics': per_class_metrics,
                'confusion_matrix': confusion_matrix(true_classes, predicted_classes).tolist(),
                'class_names': class_names,
                'total_samples': len(true_classes),
                'timestamp': datetime.now().isoformat()
            })

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_classes': true_classes,
            'predicted_classes': predicted_classes,
            'predictions': predictions,
            'class_names': class_names
        }

    def plot_confusion_matrix(self, y_true, y_pred, class_names, normalize=False):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        save_path = self.config.LOGS_DIR / f'confusion_matrix{"_normalized" if normalize else ""}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_per_class_metrics(self, y_true, y_pred, class_names):
        """Calculate metrics for each class"""
        report_dict = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        per_class_df = pd.DataFrame(report_dict).transpose()
        per_class_df = per_class_df.iloc[:-3]  # Remove avg rows

        # Plot per-class metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Precision
        axes[0, 0].bar(per_class_df.index, per_class_df['precision'])
        axes[0, 0].set_title('Precision per Class', fontweight='bold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Recall
        axes[0, 1].bar(per_class_df.index, per_class_df['recall'])
        axes[0, 1].set_title('Recall per Class', fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # F1-Score
        axes[1, 0].bar(per_class_df.index, per_class_df['f1-score'])
        axes[1, 0].set_title('F1-Score per Class', fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Support (number of samples)
        axes[1, 1].bar(per_class_df.index, per_class_df['support'])
        axes[1, 1].set_title('Support per Class', fontweight='bold')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save plot
        save_path = self.config.LOGS_DIR / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return per_class_df.to_dict()

    def plot_roc_curves(self, y_true, y_pred_proba, class_names, n_classes):
        """Plot ROC curves for multi-class classification"""
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        plt.figure(figsize=(12, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Multi-Class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Save plot
        save_path = self.config.LOGS_DIR / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return roc_auc

    def analyze_misclassifications(self, test_generator, predictions, true_classes, predicted_classes):
        """Analyze misclassified samples"""
        # Get misclassified indices
        misclassified_indices = np.where(true_classes != predicted_classes)[0]

        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return

        print(f"Number of misclassifications: {len(misclassified_indices)}")
        print(f"Misclassification rate: {len(misclassified_indices)/len(true_classes)*100:.2f}%")

        # Create misclassification analysis
        misclass_analysis = []
        class_names = list(test_generator.class_indices.keys())

        for idx in misclassified_indices[:20]:  # Analyze first 20 misclassifications
            true_class = class_names[true_classes[idx]]
            pred_class = class_names[predicted_classes[idx]]
            confidence = np.max(predictions[idx])

            misclass_analysis.append({
                'sample_index': idx,
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence,
                'true_class_confidence': predictions[idx][true_classes[idx]]
            })

        # Convert to DataFrame for analysis
        misclass_df = pd.DataFrame(misclass_analysis)

        print("\nTop 10 Misclassifications (by confidence):")
        print(misclass_df.nlargest(10, 'confidence')[['true_class', 'predicted_class', 'confidence']])

        return misclass_df

    def save_evaluation_results(self, results):
        """Save evaluation results to JSON file"""
        save_path = self.config.LOGS_DIR / 'detailed_evaluation_results.json'

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Evaluation results saved to: {save_path}")

    def generate_evaluation_report(self, test_generator):
        """Generate comprehensive evaluation report"""
        print("Generating comprehensive evaluation report...")
        print("=" * 60)

        # Perform evaluation
        eval_results = self.evaluate_on_generator(test_generator)

        # Plot ROC curves
        if len(eval_results['class_names']) <= 10:  # Only for reasonable number of classes
            self.plot_roc_curves(
                eval_results['true_classes'],
                eval_results['predictions'],
                eval_results['class_names'],
                len(eval_results['class_names'])
            )

        # Analyze misclassifications
        misclass_df = self.analyze_misclassifications(
            test_generator,
            eval_results['predictions'],
            eval_results['true_classes'],
            eval_results['predicted_classes']
        )

        # Generate summary report
        report_summary = f"""
MULTI-CLASS IMAGE CLASSIFICATION EVALUATION REPORT
==================================================

Model Performance:
- Accuracy: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)
- Precision: {eval_results['precision']:.4f}
- Recall: {eval_results['recall']:.4f}
- F1-Score: {eval_results['f1_score']:.4f}

Dataset Information:
- Total test samples: {len(eval_results['true_classes'])}
- Number of classes: {len(eval_results['class_names'])}
- Classes: {', '.join(eval_results['class_names'])}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # Save report
        report_path = self.config.LOGS_DIR / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_summary)

        print(report_summary)
        print(f"Full evaluation report saved to: {report_path}")

        return eval_results

def main():
    """Main evaluation function"""
    config = Config()

    # Load model (modify path as needed)
    model_path = config.get_model_path()

    if not Path(model_path).exists():
        print(f"Model not found at: {model_path}")
        print("Please train the model first or provide correct model path.")
        return

    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, config)

    # Load test data (you need to provide test data generator)
    from data_preprocessing import prepare_data_for_training
    _, _, test_gen, _ = prepare_data_for_training(config)

    if test_gen is None:
        print("No test data available for evaluation.")
        return

    # Generate evaluation report
    evaluator.generate_evaluation_report(test_gen)

if __name__ == "__main__":
    main()
