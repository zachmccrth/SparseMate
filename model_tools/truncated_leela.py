import numpy as np
import torch.nn as nn
import onnx2torch
import torch

from leela_interp.core.leela_board import LeelaBoard

class TruncatedModel(nn.Module):
    """
    Creates the submodule to peek into the residual stream. This is faster than what NNsight can achieve
    """

    def __init__(self, onnx_model_path, layer, device=torch.device("cpu")):
        super(TruncatedModel, self).__init__()

        self.device = device

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

        # Step 3: Create a new GraphModule
        self.submodule = torch.fx.GraphModule(lc0, new_graph).to(self.device)

        # Won't be trained, inference only
        self.requires_grad_(False)

    def forward(self, x):
        return self.submodule(x)

    def make_inputs(self, boards: list[LeelaBoard]) -> torch.Tensor:
        return torch.tensor(
            np.array([board.lcz_features(True) for board in boards]),
            dtype=torch.float32,
            device=self.device,
        )


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



