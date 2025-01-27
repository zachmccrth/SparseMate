from collections import namedtuple

import torch
import torch.autograd as autograd
import wandb
from torch import nn
from typing import Optional

from dictionary_learning.dictionary import JumpReLU
from .jumprelu import StepFunction
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)



class AttentionSeekingTrainer(nn.Module, SAETrainer):
    """
    Trains an AttentionSeeking autoencoder.

    Note this is probably bad, but maybe good
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        layer: int,
        lm_name: str,
        dict_class=JumpReLU,
        seed: Optional[int] = None,
        # TODO: What's the default lr use in the paper?
        lr: float = 7e-5,
        bandwidth: float = 0.001,
        sparsity_penalty: float = 1.0,
        warmup_steps: int = 1000,  # lr warmup period at start of training and after each resample
        sparsity_warmup_steps: Optional[int] = 2000,  # sparsity warmup period at start of training
        decay_start: Optional[int] = None,  # decay learning rate after this many steps
        target_l0: float = 20.0,
        device: str = "cpu",
        wandb_name: str = "AttentionSeeking",
        submodule_name: Optional[str] = None,
    ):
        super().__init__()

        # TODO: Should just be args, and this should be commonised
        assert layer is not None, "Layer must be specified"
        assert lm_name is not None, "Language model name must be specified"
        self.lm_name = lm_name
        self.layer = layer
        self.submodule_name = submodule_name
        self.device = device
        self.steps = steps
        self.lr = lr
        self.seed = seed

        self.bandwidth = bandwidth
        self.sparsity_coefficient = sparsity_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.decay_start = decay_start
        self.target_l0 = target_l0

        # TODO: Better auto-naming (e.g. in BatchTopK package)
        self.wandb_name = wandb_name
        self.last_grad_norm = None

        # TODO: Why not just pass in the initialised autoencoder instead?
        self.ae = dict_class(
            activation_dim=activation_dim,
            dict_size=dict_size,
            device=device,
        ).to(self.device)

        # Parameters from the paper
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        lr_fn = get_lr_schedule(
            steps,
            warmup_steps,
            decay_start,
            resample_steps=None,
            sparsity_warmup_steps=sparsity_warmup_steps,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

        # Purely for logging purposes
        self.dead_feature_threshold = 10_000_000
        self.num_tokens_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
        self.dead_features = -1
        self.logging_parameters = ["dead_features"]


    def loss(self, x: torch.Tensor, step: int, logging=False, **_):
        sparsity_scale = self.sparsity_warmup_fn(step)
        x = x.to(self.ae.dtype)

        f = self.ae.encode(x)


        # Dead features logging
        active_indices = f.sum(0) > 0
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices] = True
        self.num_tokens_since_fired += x.size(0)
        self.num_tokens_since_fired[active_indices] = 0
        self.dead_features = (
            (self.num_tokens_since_fired > self.dead_feature_threshold).sum().item()
        )

        recon = self.ae.decode(f)

        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        l0 = StepFunction.apply(f, self.ae.threshold, self.bandwidth).sum(dim=-1).mean()

        # sparsity_loss = (
        #     self.sparsity_coefficient * ((l0 / self.target_l0) - 1).pow(2) * sparsity_scale
        # )

        sparsity_loss = (f * self.ae.features.norm(p=2, dim=1)).sum(dim=-1).mean()
        loss = recon_loss + sparsity_scale * self.sparsity_coefficient * sparsity_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "recon", "f", "losses"])(
                x,
                recon,
                f,
                {
                    "recon_loss": recon_loss.item(),
                    "loss": loss.item(),
                    "sparsity_loss": sparsity_loss.item(),
                },
            )

    def update(self, step, x):
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()
        #
        # # We must transpose because we are using nn.Parameter, not nn.Linear
        # self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
        #     self.ae.W_dec.T, self.ae.W_dec.grad.T, self.ae.activation_dim, self.ae.dict_size
        # ).T
        self.last_grad_norm  = torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # # We must transpose because we are using nn.Parameter, not nn.Linear
        # self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
        #     self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        # ).T

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "AttentionSeekingTrainer",
            "dict_class": "AttentionSeekingAutoEncoder",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "bandwidth": self.bandwidth,
            "sparsity_penalty": self.sparsity_coefficient,
            "sparsity_warmup_steps": self.sparsity_warmup_steps,
            "target_l0": self.target_l0,
        }
