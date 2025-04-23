from abc import ABC, abstractmethod
from typing import List

import onnx
import onnx2torch

import torch

from leela_interp.core.leela_board import LeelaBoard

from dotenv import load_dotenv
import os

load_dotenv()


class DataEmbeddingMap(ABC):
    """
    An abstract class representing a map from the input data to the model activations
    """

    def __init__(self):
        pass

    @abstractmethod
    def convert_fen_to_input(self, fen: str) -> torch.Tensor:
        ...

    @abstractmethod
    def convert_fens_to_inputs(self, fens: List[str]) -> torch.Tensor:
        ...

    @abstractmethod
    def embed_data(self, data) -> torch.Tensor:
        ...


class TruncatedLeelaDataEmbeddingMap(DataEmbeddingMap):
    """
    A truncated version of the Leela Transformer Chess Model
    """

    def __init__(self, layers: int, device: torch.device = torch.device("cpu"), dtype = torch.float32, model_path = os.getenv("LC0_PATH"), cache_path = os.getenv("CACHE_DIR")):
        super().__init__()
        self.layers = layers
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.cache_path = cache_path

        self.submodule = self._load_model()


    def _load_model(self):

        cache_model = self._load_from_cache()

        original_graph: torch.fx.GraphModule = onnx2torch.convert(self.model_path)
        return cut_torch_graph(original_graph, f"encoder{self.layers}/ln2").to(self.device)

    def _load_from_cache(self):
        pass

    def convert_fen_to_input(self, fen: str) -> torch.Tensor:
        board = LeelaBoard.from_fen(fen)
        inputs = board.lcz_features()
        return torch.tensor(inputs, device=self.device).unsqueeze(0).to(self.dtype)

    def convert_fens_to_inputs(self, fens: List[str]) -> torch.Tensor:
        return torch.stack([self.convert_fen_to_input(fen) for fen in fens], dim=0)

    def embed_data(self, data) -> torch.Tensor:
        with torch.no_grad():
            return self.submodule(data)



def cut_torch_graph(original_graph: torch.fx.GraphModule, last_node_name: str):
    """
    Creates a subgraph model of the original graph up to the specified node.
    Args:
        original_graph:
        last_node_name:

    Returns:

    """
    new_graph = torch.fx.Graph()
    edges = {}
    for node in original_graph.graph.nodes:
        new_node = new_graph.node_copy(node, lambda n: edges[n])
        edges[node] = new_node

        if node.target == last_node_name:
            break

    new_graph.output(edges[node])  # Use the last node as the output of the new graph

    return torch.fx.GraphModule(original_graph, new_graph)


if __name__ == "__main__":
    data_embedding_map = TruncatedLeelaDataEmbeddingMap(layers=6,device=torch.device("cpu"), dtype=torch.float32)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    inputs = data_embedding_map.convert_fen_to_input(fen)
    print(inputs.shape)

    outputs = data_embedding_map.embed_data(inputs)
    print(outputs)

