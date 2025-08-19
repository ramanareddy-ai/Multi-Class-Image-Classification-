# src/config.py
from pathlib import Path
from tensorflow.keras import metrics as km  # use tf.keras metrics to avoid mixing APIs

class Config:
    # ----- Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    TRAIN_DIR = DATA_DIR / "train"
    VALIDATION_DIR = DATA_DIR / "validation"
    TEST_DIR = DATA_DIR / "test"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    SAVED_MODELS_DIR = MODELS_DIR / "saved"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

    # ----- Reproducibility / device
    RANDOM_SEED = 42
    USE_GPU = True               # harmless on CPU-only; code will print “No GPUs found”
    MIXED_PRECISION = False      # keep False on CPU for simplicity

    # ----- Data / image
    IMAGE_SIZE = (224, 224)      # (height, width)
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2       # used only if VALIDATION_DIR is empty

    # Basic augmentation used by ImageDataGenerator
    ROTATION_RANGE = 40
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True
    FILL_MODE = "nearest"

    # Loader hints (Keras 3 may ignore; keep conservative on macOS)
    NUM_WORKERS = 1

    # ----- Model / training
    MODEL_NAME = "custom_cnn"
    NUM_CLASSES = 2              # will be updated from the data generator
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 10

    # For one-hot labels + softmax
    LOSS_FUNCTION = "categorical_crossentropy"
    METRICS = [km.CategoricalAccuracy(name="accuracy")]  # Keras 3-safe

    # Callback monitoring
    SAVE_BEST_ONLY = True
    MONITOR_METRIC = "val_accuracy"
    MODE = "max"

    @classmethod
    def create_directories(cls):
        for d in [
            cls.DATA_DIR, cls.TRAIN_DIR, cls.VALIDATION_DIR, cls.TEST_DIR,
            cls.LOGS_DIR, cls.MODELS_DIR, cls.SAVED_MODELS_DIR, cls.CHECKPOINTS_DIR
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_path(cls, model_name=None):
        if model_name is None:
            model_name = f"{cls.MODEL_NAME}_best.h5"
        return cls.SAVED_MODELS_DIR / model_name

    @classmethod
    def get_checkpoint_path(cls):
        return cls.CHECKPOINTS_DIR / f"{cls.MODEL_NAME}_checkpoint.h5"
