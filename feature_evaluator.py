from typing import List
import sys
import numpy as np

from SAE_Models.autoencoders import AutoEncoderDirectory
from datasets.datasets_data import ChessBenchDataset
from model_tools.truncated_leela import TruncatedModel
# This one is necessary, don't let the IDE lie to you
# DON'T BE A SHEEP
from dictionary_learning.dictionary import *


# Add the main project directory to sys.path
project_dir = "/"
sys.path.append(project_dir)
from torch.utils.data import DataLoader
from tqdm import tqdm

from leela_interp.core.leela_board import LeelaBoard

import sqlite3
import torch

class BoardActivationBuffer:
    def __init__(self, data, submodule, d_submodule, device, size=20,):
        self.data = data
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.device = device
        self.size = size


        # buffer structure is size * boards_tokens * residual_dim
        self.activation_buffer = torch.empty(size, 64, self.d_submodule, device=self.device,
                                             dtype=torch.float32)

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
        self.boards_buffer = self.get_batch_boards(batch_size=self.size)
        inputs = self.submodule.make_inputs(self.boards_buffer)
        with torch.no_grad():
            hidden_states = self.submodule(inputs)
        self.activation_buffer = hidden_states.reshape(self.size, 64, self.d_submodule)
        self.idx = -1
        return None


def get_db_conn():
    conn = sqlite3.connect("/home/zachary/PycharmProjects/SparseMate/SparseMate.sqlite")
    return conn


def init_table(table_name):
    """
    Drops the table if it exists and recreates it.
    """
    # Initialize SQLite database and create table
    conn = get_db_conn()
    cursor = conn.cursor()

    # Drop the table if it exists
    drop_table = f"DROP TABLE IF EXISTS {table_name}"
    cursor.execute(drop_table)

    # Create the table again
    create_table = f"""
    CREATE TABLE {table_name} (
        fen TEXT,
        sq TEXT,
        feature INTEGER,
        value REAL
    )
    """
    cursor.execute(create_table)

    conn.commit()
    conn.close()


# Loop through buffer and insert data into the database
def write_to_db(dataset, autoencoder_config , table_name, total_boards_max, threshold=0.0001, device=torch.device("cpu")):
    """
    Takes a dataset and a model and evaluates the feature activations.
    """


    trainer_config = autoencoder_config["trainer"]
    buffer_config  = autoencoder_config["buffer"]


    init_table(table_name)

    dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)


    onnx_model_path = buffer_config["onnx_model_path"]
    # Have mercy for two config subfiles, something needs to change
    submodule_class =  trainer_config["submodule_name"]
    if submodule_class == "TruncatedModel":
        submodule: TruncatedModel = TruncatedModel(onnx_model_path, layer=trainer_config["layer"])
    else:
        raise Exception("Your lazy dev has not yet implemented submodules other than TruncatedModel")

    #This is an iterator
    activation_buffer = BoardActivationBuffer(
        data=iter(dataloader),
        submodule=submodule,
        d_submodule=buffer_config["d_submodule"],
        device=device,
        size=100
    )


    autoencoder_path = autoencoder_config["model_path"]
    autoencoder = globals()[trainer_config["dict_class"]].from_pretrained(autoencoder_path)

    # TODO separate autoencoder loading from db writing/eval
    # TODO maybe also decompose db writing and eval as well

    # Initialize database
    conn = get_db_conn()
    cursor = conn.cursor()

    # Batch insert data
    data_to_insert = []
    board_index = 0
    for board, activations in tqdm(activation_buffer, total=total_boards_max, unit=" boards"):
        assert isinstance(board, LeelaBoard)
        # Encode the batch to get the tensor
        features: torch.Tensor = autoencoder.encode(activations)
        features: np.ndarray = features.detach().cpu().numpy().astype(float)
        # Get the FEN string
        fen = board.fen()

        current_square_idx = -1
        # Apply threshold filtering using PyTorch


        square_idxs, feature_idxs = (features >= threshold).nonzero()
        for square_idx, feature_idx in zip(square_idxs, feature_idxs):
            square_idx = int(square_idx)
            feature_idx = int(feature_idx)
            if current_square_idx != square_idx:
                current_square_idx = square_idx
                sq = board.idx2sq(current_square_idx)

            value: float = float(features[square_idx, feature_idx])
            data_to_insert.append((fen, sq, feature_idx, value))
        # Commit in chunks to avoid memory overflow
        if len(data_to_insert) >= 1000:  # Adjust chunk size if needed
            cursor.executemany(f"""
                INSERT INTO {table_name} (fen, sq, feature, value)
                VALUES (?, ?, ?, ?)
            """, data_to_insert)
            data_to_insert.clear()

        board_index += 1

        if board_index > total_boards_max: break

    # Final commit for any remaining data
    if data_to_insert:
        cursor.executemany(f"""
            INSERT INTO {table_name} (fen, sq, feature, value)
            VALUES (?, ?, ?, ?)
        """, data_to_insert)

    # Create an index on the feature column
    cursor.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_feature ON {table_name} (feature);
    """)

    # Commit and close the connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_of_boards = 1000

    chessbench = ChessBenchDataset()

    autoencoder_directory = AutoEncoderDirectory()



    last_model_run_config = autoencoder_directory.get_last_created_model()
    print(last_model_run_config)
    table_name = f"RUN_{last_model_run_config['trainer']["run_name"].replace('_', '').replace(':', '')}"
    print(f"Writing the activations of {last_model_run_config['trainer']['run_name']} to {table_name} ")


    write_to_db(dataset=chessbench, table_name= table_name, autoencoder_config=last_model_run_config, total_boards_max=num_of_boards, threshold=0.001, device=device)

