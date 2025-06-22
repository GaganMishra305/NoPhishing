import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class PlotUtils:
    def __init__(self):
        sns.set_style("whitegrid") 
        plt.rcParams.update({'font.size': 12})

    def plot_training_history(self, history, save_path='training_plot.png'):
        if not history:
            print("❌ No training history available to plot.")
            return

        history_dict = history.history

        fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 
        fig.suptitle('Model Training History', fontsize=18, fontweight='bold')

        # Plot 1: Model Loss
        axes[0, 0].plot(history_dict['loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history_dict: 
            axes[0, 0].plot(history_dict['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontweight='bold', fontsize=14)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.5)

        # Plot 2: Model Accuracy
        axes[0, 1].plot(history_dict['accuracy'], label='Training Accuracy', color='green', linewidth=2)
        if 'val_accuracy' in history_dict: 
            axes[0, 1].plot(history_dict['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontweight='bold', fontsize=14)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.5)

        # Plot 3: Model Precision (Conditional)
        if 'precision' in history_dict and 'val_precision' in history_dict:
            axes[1, 0].plot(history_dict['precision'], label='Training Precision', color='purple', linewidth=2)
            axes[1, 0].plot(history_dict['val_precision'], label='Validation Precision', color='brown', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontweight='bold', fontsize=14)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Precision', fontsize=12)
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.5)
        else:
            axes[1, 0].axis('off') 
            axes[1, 0].text(0.5, 0.5, 'Precision data not available', 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=axes[1,0].transAxes, fontsize=12, color='gray')

        # Plot 4: Model Recall (Conditional)
        if 'recall' in history_dict and 'val_recall' in history_dict:
            axes[1, 1].plot(history_dict['recall'], label='Training Recall', color='cyan', linewidth=2)
            axes[1, 1].plot(history_dict['val_recall'], label='Validation Recall', color='magenta', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontweight='bold', fontsize=14)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Recall', fontsize=12)
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(True, alpha=0.5)
        else:
            axes[1, 1].axis('off') 
            axes[1, 1].text(0.5, 0.5, 'Recall data not available', 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=axes[1,1].transAxes, fontsize=12, color='gray')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        plt.close() 
        print(f"📊 Training plots saved to {save_path}")

    def plot_confusion_matrix_and_report(self, y_true, y_pred_prob, labels, save_path='confusion_metrics.png'):
        if y_true is None or y_pred_prob is None or labels is None or len(labels) == 0:
            print("❌ Insufficient data to generate confusion matrix and report.")
            return

        y_pred_binary = (y_pred_prob > 0.5).astype(int).flatten()
        y_true_flat = y_true.flatten()

        fig, axes = plt.subplots(2, 2, figsize=(18, 14)) 
        fig.suptitle('Model Performance Metrics', fontsize=18, fontweight='bold')

        # Plot 1: Unnormalized Confusion Matrix
        cm = confusion_matrix(y_true_flat, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0, 0], 
                    cbar_kws={'label': 'Counts'}) 
        axes[0, 0].set_title('Confusion Matrix (Counts)', fontweight='bold', fontsize=14)
        axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
        axes[0, 0].set_ylabel('True Label', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45) 
        axes[0, 0].tick_params(axis='y', rotation=0)  

        # Plot 2: Normalized Confusion Matrix
        cm_norm = confusion_matrix(y_true_flat, y_pred_binary, normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=labels, yticklabels=labels, ax=axes[0, 1],
                    cbar_kws={'label': 'Proportion'}) 
        axes[0, 1].set_title('Normalized Confusion Matrix (Proportions)', fontweight='bold', fontsize=14)
        axes[0, 1].set_xlabel('Predicted Label', fontsize=12)
        axes[0, 1].set_ylabel('True Label', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].tick_params(axis='y', rotation=0)

        # Plot 3: Classification Report (as text)
        axes[1, 0].axis('off') 
        report = classification_report(y_true_flat, y_pred_binary, target_names=labels, output_dict=True)

        report_text = "Classification Report:\n\n"
        report_text += f"{'':<12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n"
        for label in labels:
            if label in report:
                report_text += (
                    f"{label:<12}"
                    f"{report[label]['precision']:>10.3f}"
                    f"{report[label]['recall']:>10.3f}"
                    f"{report[label]['f1-score']:>10.3f}"
                    f"{int(report[label]['support']):>10}\n"
                )
        if 'accuracy' in report:
            report_text += f"\n{'accuracy':<12}{'':>10}{'':>10}{report['accuracy']:>10.3f}{int(report['macro avg']['support']):>10}\n"
        if 'macro avg' in report:
            report_text += (
                f"{'macro avg':<12}"
                f"{report['macro avg']['precision']:>10.3f}"
                f"{report['macro avg']['recall']:>10.3f}"
                f"{report['macro avg']['f1-score']:>10.3f}"
                f"{int(report['macro avg']['support']):>10}\n"
            )
        if 'weighted avg' in report:
            report_text += (
                f"{'weighted avg':<12}"
                f"{report['weighted avg']['precision']:>10.3f}"
                f"{report['weighted avg']['recall']:>10.3f}"
                f"{report['weighted avg']['f1-score']:>10.3f}"
                f"{int(report['weighted avg']['support']):>10}\n"
            )

        axes[1, 0].text(0.05, 0.95, report_text, fontfamily='monospace',
                        fontsize=10, transform=axes[1, 0].transAxes, 
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)) 
        axes[1, 0].set_title('Classification Report', fontweight='bold', fontsize=14)

        # Plot 4: Prediction Probability Distribution
        axes[1, 1].hist(y_pred_prob, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        axes[1, 1].set_title('Prediction Probability Distribution', fontweight='bold', fontsize=14)
        axes[1, 1].set_xlabel('Predicted Probability', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Confusion matrix and metrics saved to {save_path}")

    def plot_model_comparison(self, performance_data, save_path='model_comparison_summary.png'):
        if not performance_data:
            print("❌ No model performance data available for comparison.")
            return

        model_names = list(performance_data.keys())
        
        # Define the metrics to plot
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1-Score'
        }
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Distinct colors for metrics

        # Prepare data for plotting
        plot_data = {metric: [performance_data[model].get(metric, 0.0) for model in model_names] for metric in metrics}
        
        x = np.arange(len(model_names))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            offset = width * (i - (len(metrics) - 1) / 2)
            rects = ax.bar(x + offset, plot_data[metric], width, label=metric_labels[metric], color=colors[i])
            # Add value labels on top of the bars
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # Add labels and title
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12) # Rotate labels for readability
        ax.legend(fontsize=12)
        ax.set_ylim(0.0, 1.05) # Set Y-axis limit from 0 to 1.05 for scores
        ax.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines

        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Model comparison plot saved to {save_path}")

