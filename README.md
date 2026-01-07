# CLIPDR - CLIP for Diabetic Retinopathy Classification

This project implements a CLIP-based model for ordinal classification of diabetic retinopathy severity using the APTOS 2019 dataset.

## Project Structure

```
clipdr/
│
├── config.py           # Configuration parameters
├── data.py            # Dataset and data loading
├── models.py          # Model architectures (CLIPDR, PlainPromptLearner, TextEncoder)
├── fds.py             # Feature Distribution Smoothing module
├── optimizers.py      # Custom optimizers (RAdam, PlainRAdam, AdamW)
├── runner.py          # PyTorch Lightning training module
├── train.py           # Training script
├── test.py            # Testing script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning
- CLIP (OpenAI)
- OrdinalCLIP
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/openai/CLIP.git
git clone https://github.com/xk-huang/OrdinalCLIP.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the APTOS 2019 Blindness Detection dataset. Ensure your dataset is organized as follows:

```
data/
├── train_images/
├── val_images/
├── test_images/
├── train_1.csv
├── valid.csv
└── test.csv
```

Update the paths in `config.py` to match your dataset location.

## Usage

### Training

Train the model with default parameters:
```bash
python train.py
```

Train with custom parameters:
```bash
python train.py --batch_size 32 --max_epochs 150 --checkpoint_dir ./my_checkpoints
```

Available arguments:
- `--batch_size`: Batch size for training (default: 64)
- `--num_workers`: Number of data loading workers (default: 2)
- `--max_epochs`: Maximum number of training epochs (default: 100)
- `--checkpoint_dir`: Directory to save checkpoints (default: checkpoints/)
- `--log_every_n_steps`: Logging frequency (default: 50)

### Testing

Test the best model:
```bash
python test.py
```

Test a specific checkpoint:
```bash
python test.py --checkpoint_path checkpoints/best-model-epoch=99-val_acc_exp_metric=0.85.ckpt
```

Available arguments:
- `--batch_size`: Batch size for testing (default: 64)
- `--num_workers`: Number of data loading workers (default: 2)
- `--checkpoint_dir`: Directory containing checkpoints (default: checkpoints/)
- `--checkpoint_path`: Specific checkpoint path to test (optional)

## Configuration

All hyperparameters can be modified in `config.py`:

### Model Settings
- `CLIP_MODEL_NAME`: CLIP model variant (default: "RN50")
- `NUM_RANKS`: Number of severity classes (default: 5)
- `NUM_TOKENS_PER_RANK`: Tokens per rank in prompt learning (default: 1)
- `NUM_CONTEXT_TOKENS`: Number of context tokens (default: 10)

### Training Settings
- `BATCH_SIZE`: Training batch size (default: 64)
- `LEARNING_RATE`: Initial learning rate (default: 1e-4)
- `MAX_EPOCHS`: Maximum training epochs (default: 100)
- `MILESTONES`: Learning rate decay milestones (default: [60])
- `GAMMA`: Learning rate decay factor (default: 0.1)

### FDS Settings
- `FDS_FEATURE_DIM`: Feature dimension (default: 5)
- `FDS_KERNEL`: Smoothing kernel type (default: 'gaussian')
- `FDS_KS`: Kernel size (default: 5)
- `FDS_SIGMA`: Kernel sigma (default: 2)

## Model Architecture

The CLIPDR model consists of:

1. **PlainPromptLearner**: Learnable prompts for text encoding
2. **TextEncoder**: CLIP text encoder with learnable prompts
3. **CLIPDR**: Main model combining image and text encoders
4. **FDS**: Feature Distribution Smoothing for handling long-tailed distribution

## Loss Functions

The model uses three loss components:

1. **Cross-Entropy Loss**: Standard classification loss
2. **KL Divergence Loss**: For ordinal regression
3. **Rank Loss**: Custom ordinal loss for maintaining severity ordering

## Metrics

The model reports the following metrics:

- **Accuracy**: Classification accuracy
- **MAE**: Mean Absolute Error
- **AUC**: Area Under ROC Curve (macro, one-vs-one)
- **F1 Score**: Macro F1 score

Two prediction modes are evaluated:
- `exp`: Expected value prediction (soft labels)
- `max`: Maximum probability prediction (hard labels)

## Checkpointing

The training script automatically saves:
- Best model based on validation accuracy
- Last checkpoint for resuming training

Checkpoints are saved in the directory specified by `--checkpoint_dir`.

## Citation

If you use this code, please cite the original CLIP and OrdinalCLIP papers:

```bibtex
@article{yu2024clip,
  title={CLIP-DR: Textual Knowledge-Guided Diabetic Retinopathy Grading with Ranking-aware Prompting},
  author={Yu, Qinkai and Xie, Jianyang and Nguyen, Anh and Zhao, He and Zhang, Jiong and Fu, Huazhu and Zhao, Yitian and Zheng, Yalin and Meng, Yanda},
  journal={arXiv preprint arXiv:2407.04068},
  year={2024}
}
```

## License

This project is licensed under the MIT License.
