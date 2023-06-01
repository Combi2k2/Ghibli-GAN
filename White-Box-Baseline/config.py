import torch
import os

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"==== {DEVICE} ====")

# training hyper parameters
LEARNING_RATE = 1e-4  # could also use two lrs, one for gen and one for disc
BETAS = (0.5, 0.999)

BATCH_SIZE = 64
IMAGE_SIZE = 256
CHANNELS_IMG = 3
NUM_EPOCHS = 100
NUM_WORKERS = 8

FEATURES_DISC = 32
FEATURES_GEN = 32

GAN_MODE = 'lsgan'

# checkpoints' paths
SAVE_MODEL = True
LOAD_CHECKPOINT = True
CHECKPOINT_DIR = 'checkpoints/checkpoints_selfie2anime/'

CHECKPOINT_GEN = os.path.join(CHECKPOINT_DIR, 'Generator.pt')
CHECKPOINT_DISC_TX = os.path.join(CHECKPOINT_DIR, 'Discriminator_texture.pt')
CHECKPOINT_DISC_SF = os.path.join(CHECKPOINT_DIR, 'Discriminator_surface.pt')

# Loss function design
LAMBDA_SURFACE = 1
LAMBDA_TEXTURE = 1
LAMBDA_CONTENT = 10
LAMBDA_VARIANT = 5

# Dataset
LABELS = ['Human']