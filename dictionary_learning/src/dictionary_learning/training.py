"""
Training dictionaries
"""

import json

import torch.multiprocessing as mp
import os
from queue import Empty
from typing import Optional
from contextlib import nullcontext

import torch as t
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

def new_training_logging_process(run_name, config, metric_queue):
    writer = SummaryWriter(log_dir=str(os.path.join(config["log_dir"], run_name)))
    while True:
        try:
            metric_log = metric_queue.get(timeout=1)
            if metric_log == "DONE":
                break
            for metric, value in metric_log.items():
                writer.add_scalar(metric, float(value), global_step=metric_log["step"])
        except Empty:
            continue
    writer.close()


def log_metrics(trainer, step: int, embedding: t.Tensor, log_queue: mp.Queue):
    with t.no_grad():
        log = {"step": step}

        if hasattr(trainer, "last_log"):
            log.update(trainer.last_log)

        # Fraction of variance explained
        residual = trainer.ae(embedding)
        total_variance = t.var(embedding, dim=0).sum()
        residual_variance = t.var(embedding - residual, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
        log["frac_variance_explained"] = frac_variance_explained.item()

        # Log gradient norm if available
        if hasattr(trainer, "last_grad_norm") and trainer.last_grad_norm:
            log["grad_norm"] = trainer.last_grad_norm

        # Add custom trainer-level metrics if any
        trainer_log = trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            if isinstance(value, t.Tensor):
                value = value.cpu().item()
            log[f"{name}"] = value

        if log_queue:
            log_queue.put(log)


def get_norm_factor(data, steps: int) -> float:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147
    
    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm = 0
    count = 0

    for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step > steps:
            break

        count += 1
        mean_squared_norm = t.mean(t.sum(act_BD ** 2, dim=1))
        total_mean_squared_norm += mean_squared_norm

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = t.sqrt(average_mean_squared_norm).item()

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")
    
    return norm_factor


def trainSAE(
    data,
    trainer_config: dict,
    steps: int,
    use_tensorboard:bool=False,
    save_steps:Optional[list[int]]=None,
    save_dir:Optional[str]=None,
    log_steps:Optional[int]=None,
    run_cfg:dict={},
    normalize_activations:bool=False,
    device:str="cuda",
    autocast_dtype: t.dtype = t.float32,
):
    """
    Train SAEs using the given trainers

    If normalize_activations is True, the activations will be normalized to have unit mean squared norm.
    The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference.
    This is very helpful for hyperparameter transfer between different layers and models.

    Setting autocast_dtype to t.bfloat16 provides a significant speedup with minimal change in performance.
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)


    trainer_class = trainer_config["trainer"]
    del trainer_config["trainer"]
    trainer = trainer_class(**trainer_config)

    if use_tensorboard:
        run_name = trainer_config["run_name"]
        metric_log_queue = mp.Queue()
        logging_config = trainer.config | run_cfg
        # Make sure wandb config doesn't contain any CUDA tensors
        logging_config = {k: v.cpu().item() if isinstance(v, t.Tensor) else v
                      for k, v in logging_config.items()}
        tensorboard_process = mp.Process(
            target=new_training_logging_process,
            args=(run_name, logging_config, metric_log_queue),
        )
        tensorboard_process.start()

    # make save_dir, export config
    if save_dir is not None:
        save_dir = os.path.join(save_dir, f"{trainer_config["run_name"]}")
        os.makedirs(save_dir, exist_ok=True)
        # save config
        config = {"trainer": trainer.config}
        try:
            config["buffer"] = data.config
            # TODO swallows errors, very bad
        except:
            pass
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    if normalize_activations:
        norm_factor = get_norm_factor(data, steps=100)

        trainer.config["norm_factor"] = norm_factor
        # Verify that all autoencoders have a scale_biases method
        trainer.ae.scale_biases(1.0)

    # TODO figure out what this is
    tokens_per_step = 1 if data.OUT_BATCH_SIZE_TOKENS is None else data.OUT_BATCH_SIZE_TOKENS
    if tokens_per_step == 1:
        unit = "it"
    else:
        unit = "tokens"


    # Main Training Loop
    for step, act in enumerate(tqdm(data, total=steps, unit=unit, unit_scale=tokens_per_step, smoothing=0.7)):

        act = act.to(dtype=autocast_dtype)

        if normalize_activations:
            act /= norm_factor

        if step >= steps:
            break

        # logging
        if use_tensorboard and step % log_steps == 0:
            log_metrics(
                trainer, step, act, log_queue=metric_log_queue
            )

        # saving
        if save_steps is not None and step in save_steps:
            if save_dir is not None:
                if normalize_activations:
                    # Temporarily scale up biases for checkpoint saving
                    trainer.ae.scale_biases(norm_factor)

                if not os.path.exists(os.path.join(save_dir, "checkpoints")):
                    os.mkdir(os.path.join(save_dir, "checkpoints"))

                checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                t.save(
                    checkpoint,
                    os.path.join(save_dir, "checkpoints", f"ae_{step}.pt"),
                )

                if normalize_activations:
                    trainer.ae.scale_biases(1 / norm_factor)

        # training
        with autocast_context:
            trainer.update(step, act)

    # save final SAE
    if normalize_activations:
        trainer.ae.scale_biases(norm_factor)
    if save_dir is not None:
        final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
        t.save(final, os.path.join(save_dir, "ae.pt"))

    # Signal tensorboard process to finish
    if use_tensorboard:
        metric_log_queue.put("DONE")
        tensorboard_process.join()

    return trainer.ae
