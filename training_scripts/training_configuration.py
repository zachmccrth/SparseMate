import json
import os

import torch
from dotenv import load_dotenv

from dictionary_learning.dictionary import GOGS
from my_datasets.datasets_data.chessbench.bag_data import ChessBenchDataset
from training_scripts.activation_buffer import ActivationBuffer
from training_scripts.data_embedding_map import TruncatedLeelaDataEmbeddingMap
from training_scripts.train_event_logging import TensorboardEventLogger

load_dotenv()

def load_training_config(training_config_name: str = None, training_config_dir: str = os.getenv("TRAINING_CONFIG_DIR")):
    training_config = None

    if training_config_name:
        training_config_path = os.path.join(training_config_dir, training_config_name)
        with open(training_config_path, "r") as f:
            training_config = json.load(f)
            print(f"Using training config: {training_config_name}")

    if not training_config:
        training_config = {
            'model': GOGS,
            'basis_size': 4096,
            'embedding_size': 768,
            'steps': 100_000,
            'activations_generator': TruncatedLeelaDataEmbeddingMap,
            'dataset': ChessBenchDataset,
            'activations_buffer': ActivationBuffer,
            'optimizer': torch.optim.Adam,
            'scheduler': torch.optim.lr_scheduler.StepLR,
            'criterion': torch.nn.MSELoss,
            'logger': TensorboardEventLogger,
            'lr': 5e-4,
            'buffer_size': 1000,
            'batch_size': 4096,
            'layers': 6,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'dtype': torch.float32,
        }

    return training_config