
import numpy as np
import torch.nn as nn
import onnx2torch
import torch

from leela_interp.core.leela_board import LeelaBoard

class TruncatedModel(nn.Module):
    """
    Creates the submodule to peek into the residual stream. Currently hardcoded to layer 6 until I decide to be better than that
    """
    def __init__(self, onnx_model_path, device=torch.device("cpu")):
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

            if node.target == f"encoder6/ln2":
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



if __name__ == "__main__":
    layer_n = 6  # Number of layers you want to retain
    # Load ONNX model
    onnx_model_path = "/home/zachary/PycharmProjects/leela-interp/lc0.onnx"

    board = [LeelaBoard.from_fen("8/8/8/pK3k2/P7/8/8/8 b - - 1 59")]
    torch.set_float32_matmul_precision("high")
    truncated = torch.compile(TruncatedModel(onnx_model_path))

    input_tensor = truncated.make_inputs(board)

    trunc = truncated(input_tensor)

    print(trunc.shape)
