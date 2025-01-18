import torch
import numpy as np

class GroupFeatures:
    """
    Class to group features of the SAE by their board activations so they can be interpreted.
    """
    def __init__(self,
                 sparse_auto_encoder,
                 model,
                 boards,
                 residual_layer = 6,
                 model_buffer_size = 1000,
                 specialized_feature_thresholds=None,
                 relevant_boards_for_grouping = 20):
        self.sparse_auto_encoder = sparse_auto_encoder
        self.model = model
        # Currently I assume we are only grouping enough boards such that we can pass a single list
        self.boards = boards
        self.residual_layer = residual_layer
        self.model_buffer_size = model_buffer_size
        if specialized_feature_thresholds is None:
            specialized_feature_thresholds = {"activation_threshold": 0.03, "num_boards_activated_threshold": 4}
        self.specialized_feature_thresholds = specialized_feature_thresholds
        self.relevant_boards_for_grouping = relevant_boards_for_grouping

        self.activation_scores_initialized = False
        self.boards_to_activation_scores_map = torch.empty((len(boards), 12228), dtype=torch.float32)

        self.batched_scores_function = torch.vmap(self.__activation_scores_for_residual_stream__, in_dims=0)

    def compute_boards_to_activation_scores(self):
        idx = 0
        while idx < len(self.boards):
            slice_end = min(len(self.boards), idx + self.model_buffer_size)
            with self.model.trace(self.boards[idx:slice_end]):
                residual_streams = self.model.residual_stream(self.residual_layer).output.save()
                batched_scores = self.batched_scores_function(residual_streams)
                self.boards_to_activation_scores_map[idx:slice_end] = batched_scores

            idx += self.model_buffer_size

        self.activation_scores_initialized = True

    def __activation_scores_for_residual_stream__(self, residual_stream):
        feature_activations = self.sparse_auto_encoder.encode(residual_stream)

        # Right now just get the value of the maximally activating token for each feature
        return torch.max(feature_activations, dim=0).values

    def get_specialized_features_indices(self):
        if not self.activation_scores_initialized:
            return []
        return "TODO Implement this"

    def visualize_feature(self, feature_idx):
        board_activations = self.boards_to_activation_scores_map[:,feature_idx]
        top_boards = np.argsort(board_activations)[-self.relevant_boards_for_grouping:]

        with self.model.trace(top_boards):
            residual_streams = self.model.residual_stream(self.residual_layer).output.save()

        top_boards_and_heatmaps = []
        for i in range(self.relevant_boards_for_grouping):
            board_and_heatmap_dict = {
                "board": top_boards[i],
                "heatmap": self.sparse_auto_encoder.encode(residual_streams[i])
            }
            top_boards_and_heatmaps.append(board_and_heatmap_dict)

        return top_boards_and_heatmaps
