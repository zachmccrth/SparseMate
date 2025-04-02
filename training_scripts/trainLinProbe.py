import os
import sys
from datetime import datetime

project_dir = "/home/zachary/PycharmProjects/SparseMate"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


import chess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.datasets_data import ChessBenchDataset
from leela_interp.core.leela_board import LeelaBoard
from model_tools.truncated_leela import TruncatedModel



class LinearClassifierNoBias(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


class PieceClassificationDataset(IterableDataset):
    def __init__(self, base_dataset, submodule, piece_type: int):
        super().__init__()
        self.base_dataset = base_dataset
        self.submodule = submodule
        self.piece_type = piece_type

    def __iter__(self):
        for board in self.base_dataset.__iter__():
            present_squares = board.pc_board.pieces_mask(self.piece_type, False) | board.pc_board.pieces_mask(self.piece_type, True)
            inputs = self.submodule.make_inputs([board])
            output = self.submodule(inputs)
            for i in range(64):
                yield output[i], int(bool(present_squares & chess.BB_SQUARES[i]))


def train_probe(run_config, dataloader, criterion, device):
    model_class = run_config["model_class"]
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), run_config["lr"])
    writer = SummaryWriter(log_dir=run_config["log_dir"])

    total_samples = run_config["steps"] * run_config["batch_size"]
    progress_bar = tqdm(total=total_samples, desc="Training", unit="samples")

    step = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits.view(-1), batch_y.float())
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), step)
        step += 1
        progress_bar.update(len(batch_x))  # Count by number of samples

        if step >= run_config["steps"]:
            break

    progress_bar.close()
    os.makedirs(run_config["run_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_config["run_dir"], "model.pt"))
    writer.close()
    print(f"Model saved to {os.path.join(run_config['run_dir'], 'model.pt')}")


if __name__ == "__main__":
    base_dataset = ChessBenchDataset()
    model_class = LinearClassifierNoBias
    criterion = nn.BCEWithLogitsLoss()

    batch_size = 8192
    boards_to_train_on = 100_000
    steps = (boards_to_train_on * 64) // batch_size
    layer = 2
    RESIDUAL_STREAM_DIM = 768

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{datetime.now().strftime('%m%d_%H:%M')}_LinProbe"
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
        "lr": 1e-3,
        "run_dir": run_dir,
        "log_dir": log_dir,
        "batch_size": batch_size,
    }

    submodule = TruncatedModel(
        "/home/zachary/PycharmProjects/SparseMate/lc0.onnx",
        layer=run_config["layer"],
        device=device
    )

    dataset = PieceClassificationDataset(base_dataset, submodule, chess.PAWN)
    dataloader = DataLoader(dataset=dataset, batch_size=run_config["batch_size"])

    train_probe(run_config, dataloader, criterion, device)
