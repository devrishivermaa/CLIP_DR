"""Testing script for CLIPDR model."""

import sys
import os
import glob
import argparse
import pytorch_lightning as pl

# Add OrdinalCLIP to path if needed
sys.path.append("/kaggle/working/OrdinalCLIP")

import config
from data import get_dataloaders
from models import build_clipdr_model
from runner import Runner


def test(args):
    """Main testing function."""
    
    # Set up device
    print(f"Using device: {config.DEVICE}")
    
    # Get data loaders
    print("\nLoading data...")
    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Find checkpoint
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Find the best checkpoint automatically
        checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, 'best-model-*.ckpt'))
        if checkpoint_files:
            checkpoint_path = checkpoint_files[0]
        else:
            checkpoint_path = os.path.join(args.checkpoint_dir, 'last.ckpt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    # Build model
    print("\nBuilding model...")
    model = build_clipdr_model(device=config.DEVICE)
    
    # Create runner
    runner = Runner(model, num_ranks=config.NUM_RANKS)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu' if 'cuda' in config.DEVICE else 'cpu',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Test the model
    print("\nTesting the model...")
    test_results = trainer.test(runner, test_loader, ckpt_path=checkpoint_path)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    for result in test_results:
        for key, value in result.items():
            print(f"{key}: {value:.4f}")
    print("="*50)
    
    return test_results


def main():
    """Parse arguments and run testing."""
    parser = argparse.ArgumentParser(description='Test CLIPDR model on APTOS dataset')
    
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default=config.CHECKPOINT_DIR,
                        help='Directory containing checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Specific checkpoint path to test (optional)')
    
    args = parser.parse_args()
    
    # Test the model
    test_results = test(args)
    
    print("\nTesting finished successfully!")


if __name__ == "__main__":
    main()