"""Shared defaults for reproducible FasalMitra experiments."""

SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
TEST_FRACTION = 0.2
MAX_IMAGES_PER_CLASS = 80
NONIID_OVERLAP = 0.15
LEARNING_RATE = 1e-3
CENTRAL_EPOCHS = 5
FED_ROUNDS = 10
LOCAL_EPOCHS = 2

# Each client owns these crops; the configured overlap is spread across peers.
CLIENT_CROPS = {
    "client_1": ("tomato",),
    "client_2": ("apple", "cherry", "peach"),
    "client_3": ("corn", "grape"),
    "client_4": ("potato", "pepper", "strawberry"),
    "client_5": ("blueberry", "orange", "raspberry", "soybean", "squash"),
}
