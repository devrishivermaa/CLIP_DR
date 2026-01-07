"""Configuration file for CLIPDR training on APTOS dataset."""

import torch

# Paths
TRAIN_IMG_DIR = "/kaggle/input/aptos2019/train_images/train_images"
VAL_IMG_DIR = "/kaggle/input/aptos2019/val_images/val_images"
TEST_IMG_DIR = "/kaggle/input/aptos2019/test_images/test_images"

TRAIN_CSV_PATH = "/kaggle/input/aptos2019/train_1.csv"
VAL_CSV_PATH = "/kaggle/input/aptos2019/valid.csv"
TEST_CSV_PATH = "/kaggle/input/aptos2019/test.csv"

# Model settings
CLIP_MODEL_NAME = "RN50"
NUM_RANKS = 5
NUM_TOKENS_PER_RANK = 1
NUM_CONTEXT_TOKENS = 10
RANK_TOKENS_POSITION = "tail"
INIT_CONTEXT = None
RANK_SPECIFIC_CONTEXT = False

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
BETAS = (0.9, 0.999)
MAX_EPOCHS = 100
MILESTONES = [60]
GAMMA = 0.1

# FDS settings
FDS_FEATURE_DIM = 5
FDS_BUCKET_NUM = 100
FDS_BUCKET_START = 3
FDS_START_UPDATE = 0
FDS_START_SMOOTH = 1
FDS_KERNEL = 'gaussian'
FDS_KS = 5
FDS_SIGMA = 2
FDS_MOMENTUM = 0.9

# Device
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Checkpoint settings
CHECKPOINT_DIR = 'checkpoints/'
MONITOR_METRIC = 'val_acc_exp_metric'
CHECKPOINT_MODE = 'max'

# Other settings
NUM_WORKERS = 2
LOG_EVERY_N_STEPS = 50