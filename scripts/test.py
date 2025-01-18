import torch

from dictionary_learning import AutoEncoder
from groupFeatures import GroupFeatures
from leela_interp.core.leela_board import LeelaBoard
from leela_interp.core.nnsight import Lc0sight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activation_dim = 768 # output dimension of the MLP
dictionary_size = 16 * activation_dim


ae = AutoEncoder.from_pretrained("/home/zachary/PycharmProjects/SparseMate/scripts/save_dir/lichess_puzzles_epoch_1/ae.pt", device=device)
lc0: Lc0sight = Lc0sight("/home/zachary/PycharmProjects/leela-interp/lc0.onnx", device=device)




groupFeatures = GroupFeatures(sparse_auto_encoder=ae, model =lc0,
                              boards=[LeelaBoard.from_fen("3r2k1/8/6qp/8/2P1p2P/4Q3/PB3PP1/R4RK1 b - - 0 33")],
                )

groupFeatures.compute_boards_to_activation_scores()

groupFeatures.visualize_feature(134)