# device
import torch
cuda = torch.cuda.is_available()

DEVICE = "cuda" if cuda else "cpu"
print(f"==== {DEVICE} ====")

# training hyper parameters
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BETA = (0.5, 0.999)
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 8
IMAGE_SIZE = 256
CHANNELS_IMG = 3
NUM_EPOCHS = 100
NUM_WORKERS = 8 if cuda else 0

# model
FEATURES_DISC = 16
FEATURES_GEN = 16

N_DOWNSAMPLING = 2
N_RESTNET_BLOCK = 1

EMBED_DIM = 120

# loss design
ADV_WEIGHT = 1
CYCLE_WEIGHT = 10
IDENTITY_WEIGHT = 10
CAM_WEIGHT = 1000

# checkpoints' paths
SAVE_CHECKPOINT = True
LOAD_CHECKPOINT = False
CHECKPOINT_DIR = 'checkpoints/'

# Dataset
LABELS = ['City', 'Human', 'Scenery', 'Room']