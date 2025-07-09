import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.json')

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

BATCH_SIZE = cfg["batch_size"]
EPOCHS = cfg["epochs"]
LR = cfg["lr"]
IMAGE_SIZE = cfg["image_size"]
NUM_CLASSES = cfg["num_classes"]
EARLY_STOP = cfg["early_stop"]
SEED = cfg["seed"]
NEURONS_SIZE = cfg["neurons_size"]

AVG_POOL = cfg["avg_pool"]
KERNEL_SIZE = cfg["kernel_size"]
STRIDE = cfg["stride"]
PADDING = cfg["padding"]

DATA_DIR = cfg["data_dir"]
TRAIN_DIR = f"{DATA_DIR}/training"
VAL_DIR = f"{DATA_DIR}/validation"
TEST_DIR = f"{DATA_DIR}/test"


import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")