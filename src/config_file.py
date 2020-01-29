# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3

EPOCHS = 5
BATCH_SIZE = 65
VALID_SPLIT = 25

LEARNING_RATE = 0.000146
DECAY = 0.000001
AMSGRAD = True

# 0: terrestrial - 1: drone
ROBOT_TYPE = 0

# CHECKPOINT
CK_SPACEYNET_V1 = 'spaceynet_v1_model.h5'
CK_SPACEYNET_V2 = 'spaceynet_v2_model.h5'
CK_POSENET = 'posenet_model.h5'
CK_CONTEXTUALNET = 'contextualnet_model.h5'
CK_CHOICE = CK_SPACEYNET_V1
