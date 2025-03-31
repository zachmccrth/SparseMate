import itertools
from typing import Iterable

import numpy as np
import torch.utils.data
import csv

from datasets import DatasetsAPI

from leela_interp.core.leela_board import LeelaBoard


@DatasetsAPI.register('PuzzleDataset')
class PuzzleDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path, ):
        self.device = None
        self.file_path = file_path

    def preprocess(self, row):
        return LeelaBoard.from_fen(row["FEN"])


    def get_embedding(self, boards):
        if isinstance(boards, Iterable) and not isinstance(boards, LeelaBoard):
            return np.array([board.lcz_features(True) for board in boards])
        else:
            return np.array([boards.lcz_features(True)])

    def __iter__(self):
        worker_total_num = 0
        worker_id = None
        if not torch.utils.data.get_worker_info() is None:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id

        # Create an iterator
        file_itr = csv.DictReader(open(self.file_path))

        # Map each element using the line_mapper
        mapped_itr = map(self.preprocess, file_itr)

        # Add multiworker functionality
        if worker_id is not None:
            mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)

        return mapped_itr



