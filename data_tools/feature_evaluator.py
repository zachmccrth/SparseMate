from typing import List
import sys

import numpy as np

from data_tools.bag_data import ChessBenchDataset
from dictionary_learning.dictionary import JumpReluAutoEncoder, AttentionSeekingAutoEncoder

# Add the main project directory to sys.path
project_dir = "/home/zachary/PycharmProjects/SparseMate"
sys.path.append(project_dir)
from line_profiler import profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_tools.puzzles import PuzzleDataset
from dictionary_learning.buffer import tracer_kwargs
from leela_interp.core.leela_board import LeelaBoard
from leela_interp.core.leela_nnsight import Lc0sight

import sqlite3
import torch



class BoardActivationBuffer:
    def __init__(self, data, model, submodule, d_submodule, device, size=20,):
        self.data = data
        self.transformer_model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.device = device
        self.size = size

        self.activation_buffer = torch.empty(size, 64, self.d_submodule, device=self.device,
                                             dtype=self.transformer_model.dtype)

        self.boards_buffer = []

        self.idx = 20

    def get_batch_boards(self, batch_size=None) -> List[LeelaBoard]:
        """
        Return a list of boards, size self.refresh_batch_size
        """
        if batch_size is None:
            batch_size = 20
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.boards_buffer) - 1:
            self.fill_buffer()
        self.idx += 1
        return self.boards_buffer[self.idx], self.activation_buffer[self.idx]

    def fill_buffer(self):
        with torch.no_grad():
            self.boards_buffer = self.get_batch_boards(batch_size=self.size)
            with self.transformer_model.trace(
                    self.boards_buffer,
                    **tracer_kwargs,
            ):
                activations = self.submodule.output.save()
                self.submodule.output.stop()

        self.activation_buffer = activations
        self.idx = -1
        return None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
lc0: Lc0sight = Lc0sight("/home/zachary/PycharmProjects/leela-interp/lc0.onnx", device=device)
# load autoencoder
ae = AttentionSeekingAutoEncoder.from_pretrained("/home/zachary/PycharmProjects/SparseMate/scripts/save_dir/attentionseeking/ae.pt", device=device)
layer = 6

submodule = lc0.residual_stream(layer) # layer 1 MLP
activation_dim = 768 # output dimension of the MLP
dictionary_size = 16 * activation_dim


boards_to_encode = 10000
tokens_per_step = 64

# dataset = PuzzleDataset("/home/zachary/PycharmProjects/SparseMate/datasets/lichess_db_puzzle.csv")
dataset = ChessBenchDataset()

dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)

#This is an iterator
activation_buffer = BoardActivationBuffer(
    data=iter(dataloader),
    model=lc0,
    submodule=submodule,
    d_submodule=activation_dim,
    device=device,
    size=100
)


# Initialize SQLite database and create table
db_name = f"layer_{layer}_attentionseeking_chessbench.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Create the table
cursor.execute("""
CREATE TABLE IF NOT EXISTS activations (
    fen TEXT,
    sq TEXT,
    feature INTEGER,
    value REAL
)
""")
conn.commit()

# Loop through buffer and insert data into the database
@profile
def write_to_db(db_name, max, activation_buffer, threshold=0.0001):
    # Initialize database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Batch insert data
    data_to_insert = []
    board_index = 0
    for board, activations in tqdm(activation_buffer, total=max, unit=" boards"):
        assert isinstance(board, LeelaBoard)
        # Encode the batch to get the tensor
        features: torch.Tensor = ae.encode(activations)
        features: np.ndarray = features.detach().cpu().numpy().astype(float)
        # Get the FEN string
        fen = board.fen()


        current_square = -1
        # Apply threshold filtering using PyTorch

        squares, feature_idxs = (features >= threshold).nonzero()
        for square, feature in zip(squares, feature_idxs):
            square = int(square)
            if current_square != square:
                current_square = square
                sq = board.idx2sq(current_square)
            feature = int(feature)
            value: float = float(features[square, feature])
            data_to_insert.append((fen, sq, feature, value))
        # Commit in chunks to avoid memory overflow
        if len(data_to_insert) >= 1000:  # Adjust chunk size if needed
            cursor.executemany("""
                INSERT INTO activations (fen, sq, feature, value)
                VALUES (?, ?, ?, ?)
            """, data_to_insert)
            data_to_insert.clear()

        board_index += 1

        if board_index > max: break

    # Final commit for any remaining data
    if data_to_insert:
        cursor.executemany("""
            INSERT INTO activations (fen, sq, feature, value)
            VALUES (?, ?, ?, ?)
        """, data_to_insert)

    # Create an index on the feature column
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_feature ON activations (feature);
    """)

    # Commit and close the connection
    conn.commit()
    conn.close()



write_to_db(db_name,boards_to_encode,activation_buffer, threshold=0.001)

