from collections import namedtuple
from torch import nn
from dictionary_learning.trainers.trainer import SAETrainer
from ..dictionary import GOGS, GOGS2
import torch


class GOGSTrainer(nn.Module, SAETrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        layer: int,
        lm_name: str,
        dict_class=GOGS,
        lr: float = 1e-4,
        device: str = "cpu",
        run_name: str = "GOGS2",
        dtype: torch.dtype = torch.float16,
        layers: int = 6,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.steps = steps
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.layer = layer
        self.lm_name = lm_name
        self.dict_class = dict_class
        self.lr = lr
        self.device = device
        self.run_name = run_name
        self.ae = GOGS2(basis_size=dict_size, embedding_dimensions=activation_dim, device=device, dtype=dtype, iterations=layers)
        # self.loss = torch.nn.MSELoss()
        self.logging_parameters = []
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5)


    def loss(self, x: torch.Tensor, step: int, **_):
        residual = self.ae(x)
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        row_norms = self.ae.basis_set.norm(dim=1)
        l2_penalty = ((row_norms - 1.0) ** 2).mean() * 100
        total_loss = recon_loss + l2_penalty

        log_dict = {
            "step": step,
            "recon_loss": recon_loss.item(),
            "l2_penalty": l2_penalty.item(),
            "loss": total_loss.item(),
            "min_basis_norm": row_norms.min().item(),
            "max_basis_norm": row_norms.max().item(),
        }

        return total_loss, log_dict

    def update(self, step, x):
        x = x.to(self.device)
        loss, log_dict = self.loss(x, step)

        loss.backward()
        # with torch.no_grad():
        #     B = self.ae.basis_set         # shape: (num_basis, dim)
        #     G = B.grad                    # same shape
        #     if G is not None:
        #         projection = (G * B).sum(dim=1, keepdim=True) * B  # component along B
        #         G.sub_(projection)  # remove it: G = G - projection
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.last_log = log_dict  # Save logging info for later use
        return loss.item()


    @property
    def config(self):
        return {
            "trainer_class": "GOGSTrainer",
            "dict_class": "GOGS2",
            "lr": self.lr,
            "steps": self.steps,
            "activation_dim": self.activation_dim,
            "dict_size": self.dict_size,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "run_name": self.run_name,
        }