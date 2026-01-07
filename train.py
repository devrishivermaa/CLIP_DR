"""Training script for CLIPDR model."""

import sys
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Add OrdinalCLIP to path if needed
sys.path.append("/kaggle/working/OrdinalCLIP")

import config
from data import get_dataloaders
from models import build_clipdr_model
from runner import Runner


def train(args):
    """Main training function."""
    
    # Set up device
    print(f"Using device: {config.DEVICE}")
    
    # Get data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_clipdr_model(device=config.DEVICE)
    print("Model built successfully!")
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=config.MONITOR_METRIC,
        dirpath=args.checkpoint_dir,
        filename='best-model-{epoch:02d}-{val_acc_exp_metric:.2f}',
        save_top_k=1,
        mode=config.CHECKPOINT_MODE,
        save_last=True
    )
    
    # Create runner
    runner = Runner(model, num_ranks=config.NUM_RANKS)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if 'cuda' in config.DEVICE else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.fit(runner, train_loader, val_loader)
    
    print(f"\nTraining completed! Best model saved in {args.checkpoint_dir}")
    
    return trainer, runner


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description='Train CLIPDR model on APTOS dataset')
    
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS,
                        help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=config.MAX_EPOCHS,
                        help='Maximum number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str, default=config.CHECKPOINT_DIR,
                        help='Directory to save checkpoints')
    parser.add_argument('--log_every_n_steps', type=int, default=config.LOG_EVERY_N_STEPS,
                        help='Logging frequency')
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train the model
    trainer, runner = train(args)
    
    print("\nTraining finished successfully!")


if __name__ == "__main__":
    main()