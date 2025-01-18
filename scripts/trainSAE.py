import sys
import os

# Add the main project directory to sys.path
project_dir = "/home/zachary/PycharmProjects/SparseMate"
sys.path.append(project_dir)


import torch
from torch.utils.data import DataLoader
from loguru import logger
from data_tools.puzzles import PuzzleDataset
from dictionary_learning.buffer import LeelaActivationBuffer, LeelaImpActivationBuffer
from leela_interp import Lc0sight
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE

# logger.add("training_logs", level="TRACE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lc0: Lc0sight = Lc0sight("/home/zachary/PycharmProjects/leela-interp/lc0.onnx", device=device)

# Train the SAE
submodule = lc0.residual_stream(6) # layer 1 MLP
activation_dim = 768 # output dimension of the MLP
dictionary_size = 16 * activation_dim

puzzle_dataset: PuzzleDataset = PuzzleDataset("/home/zachary/PycharmProjects/SparseMate/datasets/lichess_db_puzzle.csv")

dataloader = DataLoader(puzzle_dataset, batch_size=None, batch_sampler=None)

data_len_in_boards = 1_200_000 # according to the paper, there are around 1.2 million boards

boards_to_train_on = data_len_in_boards

tokens_per_step = 100

steps = (boards_to_train_on * 64)// tokens_per_step

print(f"Training on {boards_to_train_on * 64:,} tokens in {steps:,} training steps")

activation_buffer = LeelaImpActivationBuffer(
    data=iter(dataloader),
    model=lc0,
    submodule=submodule,
    d_submodule=activation_dim,
    device=str(device),
    out_batch_size= tokens_per_step,
)

trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-3,
    "steps": steps,
    "layer": 6,
    "lm_name": "leela",
    "warmup_steps": 1000,
    "sparsity_warmup_steps": 5000,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=activation_buffer,
    trainer_configs=[trainer_cfg],
    steps=steps,
    save_dir='save_dir',
    device=str(device),
)