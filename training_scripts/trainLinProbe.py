import os
import sys
from datetime import datetime

import numpy as np

project_dir = "/home/zachary/PycharmProjects/SparseMate"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


import chess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from my_datasets.datasets_data import ChessBenchDataset
from leela_interp.core.leela_board import LeelaBoard
from model_tools.truncated_leela import TruncatedModel
import numpy as np


class LinearClassifierNoBias(nn.Module):
    def __init__(self, input_dim=768, device=torch.device("cpu")):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False, device=device)

    def forward(self, x):
        return self.linear(x)


class PieceClassificationDataset(IterableDataset):
    def __init__(self, base_dataset, submodule, piece_type: int, buffer_size=10, device=torch.device("cpu")):
        super().__init__()
        self.base_dataset = base_dataset
        self.submodule = submodule
        self.piece_type = piece_type
        self.buffer_size = buffer_size
        self.device = device
        self.dataset_iterator = iter(self.base_dataset)
        self.data_buffer = None
        self.classification_buffer = None
        self.buffer_index = 0
        self.total_buffer_size = buffer_size * 64

    def fill_buffers(self):
        boards = []
        for _ in range(self.buffer_size):
            try:
                boards.append(next(self.dataset_iterator))
            except StopIteration:
                break

        if not boards:
            raise StopIteration

        present_boards = [
            board.pc_board.pieces_mask(self.piece_type, False) | board.pc_board.pieces_mask(self.piece_type, True)
            for board in boards
        ]

        inputs = self.submodule.make_inputs(boards)
        outputs = self.submodule(inputs)
        self.data_buffer = outputs.to(self.device)
        self.classification_buffer = np.array([
            [present_squares & chess.BB_SQUARES[i] for i in range(64)]
            for present_squares in present_boards
        ]).reshape(-1).astype(bool)
        self.buffer_index = 0

    def __iter__(self):
        while True:
            if self.data_buffer is None or self.buffer_index >= self.total_buffer_size:
                self.fill_buffers()

            if self.buffer_index >= len(self.classification_buffer):
                raise StopIteration


            embedding, classification = self.data_buffer[self.buffer_index], self.classification_buffer[self.buffer_index]
            self.buffer_index += 1
            yield embedding, classification

def train_probe(run_config, dataloader, criterion, device):
    model_class = run_config["model_class"]
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), run_config["lr"])
    writer = SummaryWriter(log_dir=run_config["log_dir"])

    total_steps = run_config["steps"]
    decay_start_fraction = run_config.get("lr_decay_start", 0.9)

    def linear_tail_decay_fn(step):
        decay_start = int(decay_start_fraction * total_steps)
        if step < decay_start:
            return 1.0
        elif step >= total_steps:
            return 0.0
        else:
            decay_progress = (step - decay_start) / (total_steps - decay_start)
            return 1.0 - decay_progress

    scheduler = LambdaLR(optimizer, lr_lambda=linear_tail_decay_fn)

    total_samples = total_steps * run_config["batch_size"]
    progress_bar = tqdm(total=total_samples, desc="Training", unit="samples")

    step = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits.view(-1), batch_y.float())
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], step)

        step += 1
        progress_bar.update(len(batch_x))

        if step >= total_steps:
            break

    progress_bar.close()
    os.makedirs(run_config["run_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_config["run_dir"], "model.pt"))
    writer.close()
    print(f"Model saved to {os.path.join(run_config['run_dir'], 'model.pt')}")

def run_default_training_on_piece(piece_type):
    base_dataset = ChessBenchDataset()
    model_class = LinearClassifierNoBias
    criterion = nn.BCEWithLogitsLoss()

    batch_size = 8192
    boards_to_train_on = 60_000
    steps = (boards_to_train_on * 64) // batch_size
    layer = 6
    RESIDUAL_STREAM_DIM = 768

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{datetime.now().strftime('%m%d_%H:%M')}_{chess.PIECE_NAMES[piece_type]}_LinProbe"
    run_dir = os.path.join("SAE_Models", run_name)
    log_dir = os.path.join(run_dir, "run_logs")

    run_config = {
        "boards_trained": boards_to_train_on,
        "dataset": base_dataset.__class__.__name__,
        "steps": steps,
        "layer": layer,
        "model_class": model_class,
        "lm_name": "leela",
        "run_name": run_name,
        "lr": 1e-2,
        "run_dir": run_dir,
        "log_dir": log_dir,
        "batch_size": batch_size,
        "lr_decay_start": 0.6  # Start decaying in last 1-n% of training
    }

    submodule = TruncatedModel(
        "/home/zachary/PycharmProjects/SparseMate/lc0.onnx",
        layer=run_config["layer"],
        device=device
    )

    dataset = PieceClassificationDataset(base_dataset, submodule, piece_type, device=device, buffer_size=128)
    dataloader = DataLoader(dataset=dataset, batch_size=run_config["batch_size"])

    train_probe(run_config, dataloader, criterion, device)


if __name__ == "__main__":
    piece_types = {"PAWN": 1, "KNIGHT": 2, "BISHOP": 3, "ROOK": 4, "QUEEN": 5, "KING": 6}
    run_default_training_on_piece(piece_types[sys.argv[1]])