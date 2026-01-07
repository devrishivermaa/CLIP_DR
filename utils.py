"""Utility functions for CLIPDR project."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable and total parameters in model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_model_summary(model):
    """Print model summary with parameter counts."""
    trainable, total = count_parameters(model)
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Model size: {total * 4 / (1024**2):.2f} MB (float32)")
    print("="*50 + "\n")


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def save_predictions(y_true, y_pred, y_probs, save_path):
    """
    Save predictions to CSV file.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        save_path: Path to save the CSV file
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'prob_class_0': y_probs[:, 0],
        'prob_class_1': y_probs[:, 1],
        'prob_class_2': y_probs[:, 2],
        'prob_class_3': y_probs[:, 3],
        'prob_class_4': y_probs[:, 4],
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def load_checkpoint(model, checkpoint_path):
    """
    Load model from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        model: Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded from {checkpoint_path}")
    return model


def create_experiment_dir(base_dir='experiments'):
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        exp_dir: Path to the new experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda a, b: a < b - min_delta
        else:
            self.is_better = lambda a, b: a > b + min_delta
    
    def __call__(self, score):
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def get_class_distribution(dataset):
    """
    Get class distribution in dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        dict: Class distribution
    """
    from collections import Counter
    
    labels = [dataset[i][1] for i in range(len(dataset))]
    distribution = Counter(labels)
    
    print("\nClass Distribution:")
    print("-" * 30)
    for class_id in sorted(distribution.keys()):
        count = distribution[class_id]
        percentage = 100 * count / len(labels)
        print(f"Class {class_id}: {count} samples ({percentage:.2f}%)")
    print("-" * 30)
    
    return distribution