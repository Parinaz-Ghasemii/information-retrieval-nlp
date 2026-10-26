import os
import random
import numpy as np
import torch
import logging


# --- Project Directory Constants ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# --- Data Paths (derived from DATA_DIR) ---
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw', 'sherlock_holmes')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
COMBINED_CORPUS_PATH = os.path.join(PROCESSED_DATA_DIR, 'full_corpus.txt')
VOCAB_PATH = os.path.join(PROCESSED_DATA_DIR, 'vocab.json')
SEQUENCES_PATH = os.path.join(PROCESSED_DATA_DIR, 'sequences.pkl')




# --- Helper Functions ---
def get_experiment_dir(config):
    """
    Creates and returns the path to the directory for a specific experiment.
    This directory will hold checkpoints and the config file.
    """
    experiment_dir = os.path.join(CHECKPOINT_DIR, config.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name, log_dir=LOGS_DIR):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')