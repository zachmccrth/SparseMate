from datetime import datetime
import sys
from dictionary_learning.trainers.gogs import GOGSTrainer

from dictionary_learning.trainers.jumprelu import JumpReluCoordinateTrainer

project_dir = "/home/zachary/PycharmProjects/SparseMate"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)  # Insert at the beginning to give it precedence

from model_tools.truncated_leela import TruncatedModel
from my_datasets.datasets_data import ChessBenchDataset
from dictionary_learning.dictionary import *
from dictionary_learning.buffer import LeelaImpActivationBuffer
from dictionary_learning.training import trainSAE
import torch.multiprocessing as mp
from dictionary_learning.trainers import *
import os

if __name__ == '__main__':

    
    dataset_class = ChessBenchDataset
    # Train the SAE
    trainer_class = GOGSTrainer
    autoencoder_class = GOGS

    tokens_per_step = 2**12
    boards_to_train_on = 400_000
    sparsity_warmup_boards = boards_to_train_on - 1_000

    steps = (boards_to_train_on * 64)  // tokens_per_step
    save_interval = 5_000

    layer = 2

    RESIDUAL_STREAM_DIM = 768
    autoencoder_dim = 2**2 * RESIDUAL_STREAM_DIM
    

    print(f"Training on {boards_to_train_on * 64:,} tokens ({boards_to_train_on:,} boards) in {steps:,} training steps")
    print(f"Estimated time: {(boards_to_train_on * 64 / 40_000)/60:0.2f} minutes")


    sparsity_warmup_steps = int(round(sparsity_warmup_boards * 64 / tokens_per_step,0))
    if steps < sparsity_warmup_steps:
        raise AssertionError(f"Steps: {steps} is less than sparsity_warmup_steps: {sparsity_warmup_steps}.")


    mp.set_start_method('spawn', force=True)  # Use spawn for CUDA compatibility
    torch.set_float32_matmul_precision("high") # Should get a performance improved (torch float for ampere?)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    activation_buffer = LeelaImpActivationBuffer(
        dataset_class=dataset_class,
        onnx_model_path="/home/zachary/PycharmProjects/SparseMate/lc0.onnx",
        d_submodule=RESIDUAL_STREAM_DIM,
        device=device,
        out_batch_size= tokens_per_step,
        dtype = torch.float32,
        layer=layer,
    )

    run_name= f"{datetime.now().strftime('%m%d_%H:%M')}_{trainer_class.__name__}"

    print(f"Starting run: {run_name}")

    trainer_cfg = {
        "trainer": trainer_class,
        "dict_class": autoencoder_class,
        "activation_dim": RESIDUAL_STREAM_DIM,
        "dict_size": autoencoder_dim,
        "steps": steps,
        "layer": layer,
        "lm_name": "leela",
        "run_name": run_name,
        "device": str(device),
        "lr": 5e-4,
        "dtype": torch.float32,
        "layers": 6
    }

    # train the sparse autoencoder (SAE)
    ae= trainSAE(
        data=activation_buffer,
        trainer_config=trainer_cfg,
        steps=steps,
        save_dir="/home/zachary/PycharmProjects/SparseMate/SAE_Models",
        device=str(device),
        use_tensorboard=True,
        log_steps=20,
        autocast_dtype=torch.float32,
        save_steps=[i for i in range(save_interval, steps, save_interval)],
        run_cfg={"log_dir": os.path.join("SAE_Models",os.path.join(run_name, "run_logs"))}
    )
