import torch
import pickle

from dictionary_learning import AutoEncoder
from groupFeatures import GroupFeatures
from leela_interp.core.leela_board import LeelaBoard
from leela_interp.core.nnsight import Lc0sight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ae = AutoEncoder.from_pretrained("lichess_puzzles_epoch_1/ae.pt", device=device)
lc0: Lc0sight = Lc0sight("lc0.onnx", device=device)

with open("puzzles.pkl", "rb") as f:
    puzzles = pickle.load(f)

boards = [LeelaBoard.from_puzzle(puzzles.iloc[i]) for i in range(min(10000, len(puzzles)))]

groupFeatures = GroupFeatures(sparse_auto_encoder=ae,
                              model =lc0,
                              boards=boards,
                              model_buffer_size=100,)

groupFeatures.compute_boards_to_activation_scores()
feature_134 = groupFeatures.visualize_feature(501)
for i in range(len(feature_134)):
    (feature_134[i]['board']
     .plot(feature_134[i]['heatmap'], pre_defined_max=feature_134[i]['pre_defined_max'])
     .render(filename='./output/' + str(i) + '.svg'))
