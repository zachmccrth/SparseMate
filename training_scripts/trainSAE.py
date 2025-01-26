from datetime import datetime
import sys

from dictionary_learning.trainers.attention_seeking import AttentionSeekingTrainer
from model_tools.truncated_leela import TruncatedModel

if __name__ == '__main__':
    # Add the main project directory to sys.path
    project_dir = "/home/zachary/PycharmProjects/SparseMate"
    sys.path.append(project_dir)

    # Train the SAE
    layer = 6

    DATA_LEN_IN_BOARDS = 1_200_000 # according to the paper, there are around 1.2 million boards

    buffer_size_boards = 4_000

    boards_to_train_on = buffer_size_boards * 5

    tokens_per_step = 5000

    steps = (boards_to_train_on * 64)// tokens_per_step

    sparsity_warmup_steps = 1

    print(f"Training on {boards_to_train_on * 64:,} tokens ({boards_to_train_on:,} boards) in {steps:,} training steps")
    print(f"Estimated time: {(boards_to_train_on * 64 / 29_000)/60:0.2f} minutes")

    if steps < sparsity_warmup_steps:
        raise AssertionError(f"Steps: {steps} is less than sparsity_warmup_steps: {sparsity_warmup_steps}.")

    import torch
    from datasets import ChessBenchDataset
    from dictionary_learning.dictionary import AttentionSeekingAutoEncoder
    from dictionary_learning.buffer import  LeelaImpActivationBuffer
    from dictionary_learning.training import trainSAE
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)  # Use spawn for CUDA compatibility
    torch.set_float32_matmul_precision("high") # Should get a performance improved (torch float for ampere?)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation_dim = 768 # output dimension of the MLP
    dictionary_size = 8 * activation_dim

    dataset_class = ChessBenchDataset

    activation_buffer = LeelaImpActivationBuffer(
        dataset_class=dataset_class,
        onnx_model_path="/home/zachary/PycharmProjects/leela-interp/lc0.onnx",
        d_submodule=activation_dim,
        device=device,
        out_batch_size= tokens_per_step,
        n_ctxs=buffer_size_boards,
        dtype = torch.float16
    )

    trainer = AttentionSeekingTrainer

    trainer_cfg = {
        "trainer": trainer,
        "dict_class": AttentionSeekingAutoEncoder,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "steps": steps,
        "layer": layer,
        "lm_name": "leela",
        "wandb_name": f"{trainer.__name__}_{datetime.now().strftime('%m%d_%H:%M')}".strip(':'),
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
        use_wandb=True,
        wandb_project="SparseMate",
        wandb_entity="zacharymccrth",
        log_steps=10,
        autocast_dtype=torch.float16,
    )
