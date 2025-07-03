import torch

# Configurazione generale
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
IMAGE_SIZE = 128
NUM_CLASSES = 4  # numero di classi
EARLY_STOP = 12
SEED = 42

NEURONS_SIZE = 1024 #Numero di neuroni della rete

# Percorsi dataset
DATA_DIR = "./data"
TRAIN_DIR = f"{DATA_DIR}/training"
VAL_DIR = f"{DATA_DIR}/validation"
TEST_DIR = f"{DATA_DIR}/test"

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")