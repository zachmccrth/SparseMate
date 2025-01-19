import sys

import numpy as np

# Add the main project directory to sys.path
project_dir = "/home/zachary/PycharmProjects/SparseMate"
sys.path.append(project_dir)

# Train the SAE
layer = 6

DATA_LEN_IN_BOARDS = 1_200_000 # according to the paper, there are around 1.2 million boards

boards_to_train_on = DATA_LEN_IN_BOARDS // 20

tokens_per_step = 100

steps = (boards_to_train_on * 64)// tokens_per_step

sparsity_warmup_steps = steps // 1.2

print(f"Training on {boards_to_train_on * 64:,} tokens ({boards_to_train_on:,} boards) in {steps:,} training steps")
print(f"Estimated time: {(boards_to_train_on * 64 / 9_500)/60:0.2f} minutes")

assert steps >= sparsity_warmup_steps

import torch
from torch.utils.data import DataLoader
from data_tools.puzzles import PuzzleDataset
from dictionary_learning.buffer import  LeelaImpActivationBuffer
from leela_interp import Lc0sight
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE

# logger.add("training_logs", level="TRACE")


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
)

trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "lr": 1e-4,
    "l1_penalty": 0.01,
    "steps": steps,
    "layer": layer,
    "lm_name": "leela",
    #Note, this does nothing for the standard trainer
    "warmup_steps": 1000,
    # This does tho
    "sparsity_warmup_steps": sparsity_warmup_steps,
}

# train the sparse autoencoder (SAE)
losses, recon, l1, l1_scaled,  sparsity = trainSAE(
    data=activation_buffer,
    trainer_configs=[trainer_cfg],
    steps=steps,
    save_dir='save_dir',
    device=str(device),
)

import matplotlib.pyplot as plt
start_plots = 1000
plt.plot(losses[start_plots:])
plt.title('loss')
plt.xlabel('steps')
plt.ylabel('loss')
plt.savefig("losses.png")
plt.show()
plt.close()

plt.plot(recon[start_plots:])
plt.title('reconstruction')
plt.xlabel('steps')
plt.ylabel('loss')
plt.savefig("reconstruction.png")
plt.show()
plt.close()

plt.plot(l1[start_plots:])
plt.title('L1 Sparsity Loss')
plt.xlabel('steps')
plt.ylabel('sparsity loss')
plt.savefig("sparsity_loss.png")
plt.show()
plt.close()

plt.plot(l1_scaled[start_plots:])
plt.title('L1 Sparsity Loss Scaled')
plt.xlabel('steps')
plt.ylabel('sparsity loss')
plt.savefig("l1_loss.png")
plt.show()
plt.close()

plt.plot(sparsity[start_plots:])
plt.title('Sparsity')
plt.xlabel('steps')
plt.ylabel('average number of activations per feature')
plt.savefig("sparsity.png")
plt.show()

print(f"Final Sparsity: {np.mean(sparsity[-1000:])}")
print(f"Final Reconstruction Loss: {np.mean(recon[-1000:])}")