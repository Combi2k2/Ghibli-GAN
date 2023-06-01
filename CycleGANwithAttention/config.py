# device
import torch
cuda = torch.cuda.is_available()

DEVICE = "cuda" if cuda else "cpu"
print(f"==== {DEVICE} ====")

# training hyper parameters
LEARNING_RATE = 1e-4  # could also use two lrs, one for gen and one for disc
BETA = (0.5, 0.999)
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
NUM_EPOCHS = 10
NUM_WORKERS = 8 if cuda else 0

# model
FEATURES_DISC = 64
FEATURES_GEN = 64

N_DOWNSAMPLE_GEN = 2
N_DOWNSAMPLE_DISC = 3
N_RESTNET_BLOCK = 3

EMBED_DIM = 264

# loss design
ADV_WEIGHT = 1
CYCLE_WEIGHT = 10
IDENTITY_WEIGHT = 5
CAM_WEIGHT = 800

# checkpoints' paths
SAVE_CHECKPOINT = True
LOAD_CHECKPOINT = True
CHECKPOINT_DIR = 'checkpoints/'

# Dataset
LABELS = ['Human']