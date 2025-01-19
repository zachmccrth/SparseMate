from typing import List
import sys
# Add the main project directory to sys.path
project_dir = "/home/zachary/PycharmProjects/SparseMate"
sys.path.append(project_dir)
from line_profiler import profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_tools.puzzles import PuzzleDataset
from dictionary_learning import AutoEncoder
from dictionary_learning.buffer import tracer_kwargs
from leela_interp.core.leela_board import LeelaBoard
from leela_interp.core.leela_nnsight import Lc0sight

import sqlite3
import torch



class BoardActivationBuffer():
    def __init__(self, data, model, submodule, d_submodule, device):
        self.data = data
        self.transformer_model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.device = device

        self.activation_buffer = torch.empty(64, self.d_submodule, device=self.device,
                                             dtype=self.transformer_model.dtype)

    def get_batch_boards(self, batch_size=None) -> List[LeelaBoard]:
        """
        Return a list of boards, size self.refresh_batch_size
        """
        if batch_size is None:
            batch_size = 1
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def __iter__(self):
        return self

    def __next__(self):
        with torch.no_grad():
            boards = self.get_batch_boards()
            with self.transformer_model.trace(
                    boards,
                    **tracer_kwargs,
            ):

                hidden_states = self.submodule.output.save()

                self.submodule.output.stop()

        # Flatten along the first dimension (flatten batches and boards together), we only need random residual encodings from now on
        activations = hidden_states.reshape(-1, 768)
        # TODO dont be like this
        return boards[0], activations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lc0: Lc0sight = Lc0sight("/home/zachary/PycharmProjects/leela-interp/lc0.onnx", device=device)
# load autoencoder
ae = AutoEncoder.from_pretrained("/home/zachary/PycharmProjects/SparseMate/scripts/save_dir/lichess_puzzles_epoch_1_v2/ae.pt", device=device)
layer = 6

submodule = lc0.residual_stream(layer) # layer 1 MLP
activation_dim = 768 # output dimension of the MLP
dictionary_size = 16 * activation_dim


boards_to_encode = 10
tokens_per_step = 64

puzzle_dataset = PuzzleDataset("/home/zachary/PycharmProjects/SparseMate/datasets/lichess_db_puzzle.csv")

dataloader = DataLoader(puzzle_dataset, batch_size=None, batch_sampler=None)

#This is an iterator
activation_buffer = BoardActivationBuffer(
    data=iter(dataloader),
    model=lc0,
    submodule=submodule,
    d_submodule=activation_dim,
    device=device,
)


# Initialize SQLite database and create table
db_name = f"layer_{layer}.db"
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
        # Encode the batch to get the tensor
        features: torch.Tensor = ae.encode(activations)

        # Get the FEN string
        fen = board.fen()

        # Apply threshold filtering using PyTorch
        valid_indices = (features >= threshold).nonzero(as_tuple=False)
        for idx in valid_indices:
            sq_idx, feature_idx = idx[0].item(), idx[1].item()
            value = features[sq_idx, feature_idx].item()
            sq = board.idx2sq(sq_idx)  # Map square index to board square
            data_to_insert.append((fen, sq, feature_idx, value))

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

    # Commit and close the connection
    conn.commit()
    conn.close()



write_to_db(db_name,boards_to_encode,activation_buffer)

