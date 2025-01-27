from datetime import datetime
import sys

from dictionary_learning.trainers.jumprelu import JumpReLUFunction
from model_tools.truncated_leela import TruncatedModel
import torch
from datasets import ChessBenchDataset
from dictionary_learning.dictionary import *
from dictionary_learning.buffer import LeelaImpActivationBuffer
from dictionary_learning.training import trainSAE
import torch.multiprocessing as mp
from dictionary_learning.trainers import *

if __name__ == '__main__':
    # Add the main project directory to sys.path
    project_dir = "/home/zachary/PycharmProjects/SparseMate"
    sys.path.append(project_dir)
    
    dataset_class = ChessBenchDataset
    # Train the SAE
    trainer_class = JumpReluTrainer
    autoencoder_class = JumpReLU



    tokens_per_step = 5000
    boards_to_train_on = 100_000
    sparsity_warmup_boards = 1000

    steps = (boards_to_train_on * 64)// tokens_per_step

    layer = 6

    RESIDUAL_STREAM_DIM = 768
    autoencoder_dim = 16 * RESIDUAL_STREAM_DIM
    

    print(f"Training on {boards_to_train_on * 64:,} tokens ({boards_to_train_on:,} boards) in {steps:,} training steps")
    print(f"Estimated time: {(boards_to_train_on * 64 / 29_000)/60:0.2f} minutes")


    sparsity_warmup_steps = int(round(sparsity_warmup_boards * 64 / tokens_per_step,0))
    if steps < sparsity_warmup_steps:
        raise AssertionError(f"Steps: {steps} is less than sparsity_warmup_steps: {sparsity_warmup_steps}.")


    mp.set_start_method('spawn', force=True)  # Use spawn for CUDA compatibility
    torch.set_float32_matmul_precision("high") # Should get a performance improved (torch float for ampere?)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    activation_buffer = LeelaImpActivationBuffer(
        dataset_class=dataset_class,
        onnx_model_path="/home/zachary/PycharmProjects/leela-interp/lc0.onnx",
        d_submodule=RESIDUAL_STREAM_DIM,
        device=device,
        out_batch_size= tokens_per_step,
        dtype = torch.float16
    )


    trainer_cfg = {
        "trainer": trainer_class,
        "dict_class": autoencoder_class,
        "activation_dim": RESIDUAL_STREAM_DIM,
        "dict_size": autoencoder_dim,
        "steps": steps,
        "layer": layer,
        "lm_name": "leela",
        "wandb_name": f"{trainer_class.__name__}_{datetime.now().strftime('%m%d_%H:%M')}",
        "device": str(device),
        "target_l0": 40,
        "sparsity_warmup_steps": sparsity_warmup_steps,
        "warmup_steps": sparsity_warmup_steps,
        "sparsity_penalty" : 0.1,
        "submodule_name": TruncatedModel.__name__,
    }

    # train the sparse autoencoder (SAE)
    ae= trainSAE(
        data=activation_buffer,
        trainer_configs=[trainer_cfg],
        steps=steps,
        save_dir="/home/zachary/PycharmProjects/SparseMate/SAE_Models",
        device=str(device),
        use_tensorboard=True,
        wandb_project="SparseMate",
        wandb_entity="zacharymccrth",
        log_steps=10,
        autocast_dtype=torch.float16,
    )
