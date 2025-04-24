import torch
from data_embedding_map import DataEmbeddingMap, TruncatedLeelaDataEmbeddingMap
from my_datasets import DatasetsAPI


class ActivationBuffer:
    """
    A buffer to hold examined model activations for training
    """
    def __init__(self, dataset: torch.utils.data.Dataset, activations_generator: DataEmbeddingMap,
                 buffer_size: int, device ='cpu', dtype = torch.float32):
        self.dataset = dataset
        # batch_size = None necessary to make sure that PyTorch doesn't call collate
        self.datasource: torch.utils.data.DataLoader = torch.utils.data.DataLoader(chessbench_dataset,
                                                                                   batch_size=None, batch_sampler=None)
        self.data_iter= iter(self.datasource)
        self.embedding_map = activations_generator
        self.buffer = torch.empty((buffer_size, activations_generator.activation_dimensions), dtype=dtype, device = device)
        self.buffer_size_data = buffer_size
        self.dtype = dtype
        self.index = None

        self._fill_buffer()


    def _fill_buffer(self) -> None:
        """
        Reset and refill the buffer
        """
        input_data =self.embedding_map.convert_list_to_input([next(self.data_iter) for _ in range(self.buffer_size_data)])
        self.buffer = self.embedding_map.embed_data(input_data)
        self.index = 0


    def load_next_data(self, n_samples) -> torch.Tensor:
        """
        Loads n samples from the buffer, reloading if necessary
        """
        if self.buffer_size_data < (self.index + n_samples):
            overflow = self.index + n_samples - self.buffer_size_data
            first_part = self.load_next_data(n_samples - overflow)
            self._fill_buffer()
            return torch.concatenate((first_part, self.load_next_data(overflow)), dim=0)
        next_data = self.buffer[self.index:(self.index + n_samples)]
        self.index += n_samples
        return next_data

if __name__ == '__main__':
    chessbench_dataset = DatasetsAPI.get("ChessBenchDataset")
    embedding_map = TruncatedLeelaDataEmbeddingMap(2)
    buffer = ActivationBuffer(chessbench_dataset, embedding_map, 10)
    for i in range(12):
        print(buffer.load_next_data(2))