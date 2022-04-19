from yacs.config import CfgNode as CN

_C = CN()

_C.SETTINGS = CN()
_C.SETTINGS.DEVICE = 'cuda'

_C.DATASET = CN()
_C.DATASET.DATA_PATH = 'data/ibm_debater/full'
_C.DATASET.LOAD_AUDIO = False
_C.DATASET.LOAD_TEXT = True
_C.DATASET.TOKENIZER = 'distilbert-base-uncased'
_C.DATASET.CHUNK_LENGTH = 5
_C.DATASET.SMALL_VERSION = False

_C.DATASET.LOADER = CN()
_C.DATASET.LOADER.BATCH_SIZE = 16
_C.DATASET.LOADER.DROP_LAST = True

_C.MODEL = CN()
_C.MODEL.NAME = 'text'

_C.MODEL.TEXT = CN()
_C.MODEL.TEXT.N_TRAINABLE_LAYERS = 2
_C.MODEL.TEXT.CLASSIFY = True
_C.MODEL.TEXT.DROPOUT_VALUES = [0.3, 0.3]
_C.MODEL.TEXT.PRE_CLASSIFIER = True
_C.MODEL.TEXT.DISTILBERT = 'distilbert-base-uncased'

_C.MODEL.AUDIO = CN()
_C.MODEL.AUDIO.DOWNSAMPLER_OUT_DIM = 32
_C.MODEL.AUDIO.BILSTM_HIDDEN_SIZE = 256
_C.MODEL.AUDIO.N_TRAINABLE_LAYERS = 2
_C.MODEL.AUDIO.CLASSIFY = False
_C.MODEL.AUDIO.DROPOUT_VALUES = [0.3, 0.3, 0.3]
_C.MODEL.AUDIO.PRE_CLASSIFIER = True

_C.MODEL.MULTIMODAL = CN()
_C.MODEL.MULTIMODAL.DROPOUT_VALUES = [0.3]

_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.OPTIMIZER_ARGS = CN(new_allowed=True)
_C.TRAIN.LR = 2e-5
_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)

_C.TRAIN.EARLY_STOPPING = CN()
_C.TRAIN.EARLY_STOPPING.PATIENCE = 3

_C.TRAIN.EPOCHS = 20

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()