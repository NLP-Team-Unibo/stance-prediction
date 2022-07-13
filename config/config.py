from yacs.config import CfgNode as CN

_C = CN()

_C.SETTINGS = CN()
_C.SETTINGS.DEVICE = 'cuda'

_C.DATASET = CN()
_C.DATASET.DATA_PATH = 'data/ibm_debater/full'
_C.DATASET.LOAD_AUDIO = False
_C.DATASET.LOAD_TEXT = True
_C.DATASET.TOKENIZER = 'distilbert-base-uncased'
_C.DATASET.CHUNK_LENGTH = 15
_C.DATASET.SMALL_VERSION = False
_C.DATASET.SAMPLE_CUT_TYPE = 'first' # 'first', 'last', 'both'

_C.DATASET.LOADER = CN()
_C.DATASET.LOADER.BATCH_SIZE = 16
_C.DATASET.LOADER.DROP_LAST = False
_C.DATASET.LOADER.NUM_WORKERS = 0


_C.MODEL = CN()
_C.MODEL.NAME = 'text'

_C.MODEL.TEXT = CN()
_C.MODEL.TEXT.N_TRAINABLE_LAYERS = 2
_C.MODEL.TEXT.CLASSIFY = True
_C.MODEL.TEXT.DROPOUT_VALUES = [0.3, 0.3]
_C.MODEL.TEXT.PRE_CLASSIFIER = True
_C.MODEL.TEXT.DISTILBERT = 'distilbert-base-uncased'

_C.MODEL.AUDIO = CN()
_C.MODEL.AUDIO.N_TRANSFORMERS = 4
_C.MODEL.AUDIO.N_TRAINABLE_LAYERS = 4
_C.MODEL.AUDIO.CLASSIFY = False
_C.MODEL.AUDIO.DROPOUT_VALUES = [0.3, 0.3]
_C.MODEL.AUDIO.PRE_CLASSIFIER = True

_C.MODEL.MULTIMODAL = CN()
_C.MODEL.MULTIMODAL.DROPOUT_VALUES = [0.3]

_C.MODEL.MULTIMODAL.LOAD_TEXT_CHECKPOINT = False
_C.MODEL.MULTIMODAL.LOAD_AUDIO_CHECKPOINT = False
_C.MODEL.MULTIMODAL.TEXT_CHECKPOINT_PATH = ''
_C.MODEL.MULTIMODAL.AUDIO_CHECKPOINT_PATH = ''
_C.MODEL.MULTIMODAL.FREEZE_TEXT = False
_C.MODEL.MULTIMODAL.FREEZE_AUDIO = False

_C.MODEL.MULTIMODAL.CROSS = CN()
_C.MODEL.MULTIMODAL.CROSS.USE = True
_C.MODEL.MULTIMODAL.CROSS.TYPE = 'audio2text'
_C.MODEL.MULTIMODAL.CROSS.POOL = 'avg'

_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.OPTIMIZER_ARGS = CN(new_allowed=True)
_C.TRAIN.LR = 2e-5
_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)
_C.TRAIN.SAVE_CHECKPOINT = False
_C.TRAIN.CHECKPOINT_PATH = ''

_C.TRAIN.EARLY_STOPPING = CN()
_C.TRAIN.EARLY_STOPPING.PATIENCE = 3

_C.TRAIN.EPOCHS = 20

def get_cfg_defaults():
    """
    Returns a clone of the yacs CfgNode object with default values for the project, so that the original defaults 
    will not be altered.
    """
    return _C.clone()

def save_cfg_default():
    """Save in a YAML file the default version of the configuration file, in order to provide a template to be modified."""
    with open('config/default.yaml', 'w') as f:
        f.write(_C.dump())
        f.flush()
        f.close()

def save_cfg_text_default():
    """Save in a YAML file a template configuration file for training the TextModel."""
    cfg = _C.clone()
    with open('config/text.yaml', 'w') as f:
        del cfg['MODEL']['AUDIO']
        del cfg['MODEL']['MULTIMODAL']
        f.write(cfg.dump())
        f.flush()
        f.close()

def save_cfg_audio_default():
    """Save in a YAML file a template configuration file for training the AudioModel."""
    cfg = _C.clone()
    cfg.DATASET.LOADER.BATCH_SIZE = 8
    cfg.DATASET.LOAD_AUDIO = True
    cfg.DATASET.LOAD_TEXT = False
    cfg.MODEL.AUDIO.CLASSIFY = True
    cfg.MODEL.NAME = 'audio'
    with open('config/audio.yaml', 'w') as f:
        del cfg['MODEL']['TEXT']
        del cfg['MODEL']['MULTIMODAL']
        f.write(cfg.dump())
        f.flush()
        f.close()

def save_cfg_multimodal_default():
    """Save in a YAML file a template configuration file for training the MultimodalModel."""
    cfg = _C.clone()
    cfg.DATASET.LOADER.BATCH_SIZE = 8
    cfg.DATASET.LOAD_AUDIO = True
    cfg.MODEL.NAME = 'multimodal'
    cfg.MODEL.TEXT.CLASSIFY = False
    cfg.MODEL.AUDIO.CLASSIFY = False
    with open('config/multimodal.yaml', 'w') as f:
        f.write(cfg.dump())
        f.flush()
        f.close()

if __name__ == '__main__':
    """Automatically save the template version of the config files when config.py is executed as a script."""
    save_cfg_default()
    save_cfg_text_default()
    save_cfg_audio_default()
    save_cfg_multimodal_default()