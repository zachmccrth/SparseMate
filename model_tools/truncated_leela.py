from typing import List, Optional

import numpy as np
import torch.nn as nn
import onnx2torch
import torch
import os

from leela_interp.core.leela_board import LeelaBoard

class TruncatedModel(nn.Module):
    """
    Creates the submodule to peek into the residual stream. This is faster than what NNsight can achieve
    """

    def __init__(self, onnx_model_path, layer, device=torch.device("cpu"), cache=True, compile=True):
        super(TruncatedModel, self).__init__()

        self.device = device
        self._example_inputs = (self.make_inputs([LeelaBoard.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")]),)
        self.layer = layer
        self.onnx_model_path = onnx_model_path

        directory = self._check_cache()

        if cache & (directory is not None):
            self.submodule = self.read_from_cache(os.path.join(directory, "onnx_model.onnx"))
        else:
            self.submodule = self.extract_from_onnx(layer, onnx_model_path)

        # Won't be trained, inference only
        self.requires_grad_(False)

    def extract_from_onnx(self, layer, onnx_model_path) -> torch.fx.GraphModule:
        # Convert to PyTorch model
        lc0: torch.fx.GraphModule = onnx2torch.convert(onnx_model_path)
        new_graph = torch.fx.Graph()
        env = {}  # A mapping of original nodes to new nodes in the subgraph
        for node in lc0.graph.nodes:
            # Copy the node to the new graph
            new_node = new_graph.node_copy(node, lambda n: env[n])
            env[node] = new_node

            if node.target == f"encoder{layer}/ln2":
                break  # Stop copying nodes once you reach the split point
        # Specify the new output
        new_graph.output(env[node])  # Use the last node as the output of the new graph
        return torch.fx.GraphModule(lc0, new_graph).to(self.device)

    def _get_cache_dir(self) -> str:
        return "/home/zachary/PycharmProjects/SparseMate/cache/TruncatedModelCache"

    def forward(self, x):
        return self.submodule(x)

    def make_inputs(self, boards: list[LeelaBoard]) -> torch.Tensor:
        return torch.tensor(
            np.array([board.lcz_features(True) for board in boards]),
            dtype=torch.float32,
            device=self.device,
        )

    def save_to_cache(self) -> None:
        model_dir = f"/home/zachary/PycharmProjects/SparseMate/cache/TruncatedModelCache/layer_{self.layer}"
        os.mkdir(model_dir)
        onnx_program = torch.onnx.export(self.submodule, self._example_inputs, dynamo=True)
        onnx_program.optimize()
        onnx_program.save(f"~/PycharmProjects/SparseMate/cache/TruncatedModelCache/layer_{self.layer}/onnx_model.onnx")

    def _check_cache(self) -> Optional[str]:
        """
        Checks cache for presence of identical saved model, returns directory or None
        """
        directory_list: List[str] = os.listdir(self._get_cache_dir())
        for directory in directory_list:
            if self._is_model_equivalent(directory):
                return directory
        return None

    def _is_model_equivalent(self, path_to_model_directory: str) -> bool:
        # TODO CHANGE TO CONFIG/METADATA
        return path_to_model_directory.endswith(f"layer_{self.layer}")

    def read_from_cache(self, path_to_cached) -> torch.fx.GraphModule:
        """
        Returns the required submodule
        """
        submodule_path = os.path.join(path_to_cached, "onnx_model.onnx")
        return  onnx2torch.convert(submodule_path)

    def compile(self):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        flag = NetworkDefinitionCreationFlag.STRONGLY_TYPED
        network = builder.create_network(flag)


        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(self.onnx_model_path)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))

        if not success:
            pass  # Error handling code here

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MiB
        serialized_engine = builder.build_serialized_network(network, config)
        with open(“sample.engine”, “wb”) as f:
            f.write(serialized_engine)


class SingleLayer(nn.Module):
    def __init__(self, onnx_model_path, layer, device=torch.device("cpu")):
        super(SingleLayer, self).__init__()
        self.device = device

        lc0: torch.fx.GraphModule = onnx2torch.convert(onnx_model_path)
        new_graph = torch.fx.Graph()
        env = {}  # A mapping of original nodes to new nodes in the subgraph

        # Create a new input node (this will replace the input to encoder{layer})
        input_node = new_graph.placeholder("new_input")
        def get_or_copy(n):
            """Copy nodes, but replace inputs to layer n with new inputs."""
            if n not in env:
                if n.op == "placeholder":
                    # Replace the original input with our new input
                    env[n] = input_node
                else:
                    env[n] = new_graph.node_copy(n, lambda x: get_or_copy(x))
            return env[n]

        # Copy only nodes belonging to layer n and beyond
        for node in lc0.graph.nodes:
            if isinstance(node.target, str) and node.target.startswith(f"encoder{layer}"):
                env[node] = new_graph.node_copy(node, get_or_copy)
                last_node = env[node]

        new_graph.output(last_node)
        self.submodule = torch.fx.GraphModule(lc0, new_graph).to(self.device)

        # Won't be trained, inference only
        self.requires_grad_(False)

    def forward(self, x):
        return self.submodule(x)



if __name__ == "__main__":
    layer_n = 6  # Layer to extract
    onnx_model_path = "/home/zachary/PycharmProjects/SparseMate/lc0.onnx"

    model = SingleLayer(onnx_model_path=onnx_model_path, layer=layer_n, device=torch.device("cpu"))

    dummy = torch.randn(64, 768)  # Match expected input shape
    output = model(dummy)  # Get the model's actual output
    for node in model.submodule.graph.nodes:
        if node.op == "placeholder":
            print(f"Input Node: {node.name}, Shape: {node.meta.get('tensor_meta', 'Unknown')}")



