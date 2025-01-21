from datetime import datetime
import sys

import numpy as np

from dictionary_learning.dictionary import JumpReluAutoEncoder

# Add the main project directory to sys.path
project_dir = "/home/zachary/PycharmProjects/SparseMate"
sys.path.append(project_dir)

# Train the SAE
layer = 6

DATA_LEN_IN_BOARDS = 1_200_000 # according to the paper, there are around 1.2 million boards

buffer_size_boards = 10_000

boards_to_train_on = (buffer_size_boards - 10) * 40

tokens_per_step = 2000

steps = (boards_to_train_on * 64)// tokens_per_step

sparsity_warmup_steps = 100

print(f"Training on {boards_to_train_on * 64:,} tokens ({boards_to_train_on:,} boards) in {steps:,} training steps")
print(f"Estimated time: {(boards_to_train_on * 64 / 9_000)/60:0.2f} minutes")

if steps < sparsity_warmup_steps:
    raise AssertionError(f"Steps: {steps} is less than sparsity_warmup_steps: {sparsity_warmup_steps}.")

import torch
from torch.utils.data import DataLoader
from data_tools.puzzles import PuzzleDataset
from dictionary_learning.buffer import  LeelaImpActivationBuffer
from leela_interp import Lc0sight
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer, JumpReluTrainer
from dictionary_learning.training import trainSAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lc0: Lc0sight = Lc0sight("/home/zachary/PycharmProjects/leela-interp/lc0.onnx", device=device)

submodule = lc0.residual_stream(layer) # layer 1 MLP
activation_dim = 768 # output dimension of the MLP
dictionary_size = 16 * activation_dim

puzzle_dataset: PuzzleDataset = PuzzleDataset("/home/zachary/PycharmProjects/SparseMate/datasets/lichess_db_puzzle.csv")

dataloader = DataLoader(puzzle_dataset, batch_size=None, batch_sampler=None)


activation_buffer = LeelaImpActivationBuffer(
    data=iter(dataloader),
    model=lc0,
    submodule=submodule,
    d_submodule=activation_dim,
    device=str(device),
    out_batch_size= tokens_per_step,
    n_ctxs=buffer_size_boards
)

trainer = JumpReluTrainer

trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "trainer": trainer,
    "dict_class": JumpReluAutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-4,
    "l1_penalty": 0.01,
    "steps": steps,
    "layer": layer,
    "lm_name": "leela",
    "wandb_name": f"{trainer.__name__}_{datetime.now().strftime('%m%d_%H:%M')}",
    "device": str(device),
    "target_l0": 40,
    "sparsity_warmup_steps": sparsity_warmup_steps,
}

# train the sparse autoencoder (SAE)
ae= trainSAE(
    data=activation_buffer,
    trainer_configs=[trainer_cfg],
    steps=steps,
    save_dir='save_dir',
    device=str(device),
    use_wandb=True,
    wandb_project="SparseMate",
    wandb_entity="zacharymccrth",
    log_steps=500
)
