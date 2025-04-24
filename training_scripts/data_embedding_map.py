import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import onnx2torch
import torch
from leela_interp.core.leela_board import LeelaBoard
from loguru import logger
from dotenv import load_dotenv
import os
from loguru import logger

load_dotenv()


class DataEmbeddingMap(ABC):
    """
    An abstract class representing a map from the input data to the model activations
    """

    def __init__(self):
        pass

    @abstractmethod
    def convert_to_input(self, fen: str) -> torch.Tensor:
        ...

    @abstractmethod
    def convert_list_to_input(self, fens: List[str]) -> torch.Tensor:
        ...

    @abstractmethod
    def embed_data(self, data) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def activation_dimensions(self) -> int:
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
        self._ACTIVATION_DIMENSIONS = 768
        self._CACHE_PATH =  os.path.join(cache_path, "TruncatedLeelaCache")
        self._CONFIG_FILE_NAME = 'config.json'
        self._SUBMODULE_FILE_NAME = 'model.pt'
        self._CACHE_DIR = os.path.join(self._CACHE_PATH, f"{self.__class__.__name__}_{layers}")

        self.submodule: torch.nn.Module = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """
        Loads a model, either by constructing one from the original ONNX graph, or a precached model
        """
        model = self._load_from_cache()
        if not model:
            original_graph: torch.fx.GraphModule = onnx2torch.convert(self.model_path)
            model = cut_torch_graph(original_graph, f"encoder{self.layers}/ln2").to(self.device)
            self._save_to_cache(model)
        return model

    def _save_to_cache(self, model) -> None:
        os.makedirs(self._CACHE_DIR, exist_ok=True)
        torch.save(model, os.path.join(self._CACHE_DIR, self._SUBMODULE_FILE_NAME))

        with open(os.path.join(self._CACHE_DIR, self._CONFIG_FILE_NAME), "w") as f:
            json.dump(self.config, f, indent=2)


    def _load_from_cache(self) -> Optional[torch.nn.Module]:
        """
        Searches the model cache and loads model if a match is found
        """
        cached_model_dir = self._search_cache()
        if not cached_model_dir:
            return None

        cached_model_path = os.path.join(cached_model_dir, self._SUBMODULE_FILE_NAME)

        model = torch.load(cached_model_path, weights_only=False)

        return model

    def _search_cache(self) -> Optional[str]:
        """
        Returns the top level directory for a previously saved model
        """
        if not os.path.isdir(self._CACHE_PATH):
            logger.warning(f"{self._CACHE_PATH} does not exist! Cache is assumed to be empty")
            return None

        # os.walk returns an iterator of tuples (dirpath, dirnames, filenames)
        for root, dirs, files in os.walk(self._CACHE_PATH):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                # Check contents of each directory
                for file in os.listdir(dir_path):
                    if file == self._CONFIG_FILE_NAME:
                        config_path = os.path.join(dir_path, file)
                        with open(config_path) as f:
                            config = json.load(f)
                            if config == self.config:
                                return dir_path
        return None

    def convert_to_input(self, board: LeelaBoard) -> torch.Tensor:
        return torch.tensor(board.lcz_features(), device=self.device).unsqueeze(0).to(self.dtype)

    def convert_list_to_input(self, boards: List[LeelaBoard]) -> torch.Tensor:
        return torch.concatenate([self.convert_to_input(board) for board in boards], dim=0)

    def embed_data(self, data) -> torch.Tensor:
        with torch.no_grad():
            return self.submodule(data)

    @property
    def activation_dimensions(self) -> int:
        return self._ACTIVATION_DIMENSIONS

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "base_model": "Lc0Model",
            "activation_dim": self._ACTIVATION_DIMENSIONS,
            "dtype": str(self.dtype),
            "layers": self.layers,
            "model_path": self.model_path,
        }


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
    inputs = data_embedding_map.convert_to_input(fen)
    print(inputs.shape)

    outputs = data_embedding_map.embed_data(inputs)
    print(outputs)