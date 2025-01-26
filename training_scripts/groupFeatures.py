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
                 feature_dimensions = 12288,
                 model_buffer_size = 1000,
                 specialized_feature_thresholds=None,
                 relevant_boards_for_grouping = 20):
        self.sparse_auto_encoder = sparse_auto_encoder
        self.model = model
        # Currently I assume we are only grouping enough boards such that we can pass a single list
        self.boards = boards
        self.residual_layer = residual_layer
        self.feature_dimensions = feature_dimensions
        self.model_buffer_size = model_buffer_size
        if specialized_feature_thresholds is None:
            specialized_feature_thresholds = {"activation_threshold": 0.03, "num_boards_activated_threshold": 4}
        self.specialized_feature_thresholds = specialized_feature_thresholds
        self.relevant_boards_for_grouping = relevant_boards_for_grouping

        self.activation_scores_initialized = False
        self.boards_to_activation_scores_map = torch.empty((len(boards), self.feature_dimensions), dtype=torch.float32, requires_grad=False)

    @torch.no_grad()
    def compute_boards_to_activation_scores(self):
        idx = 0
        while idx < len(self.boards):
            print('Index = ' + str(idx))
            slice_end = min(len(self.boards), idx + self.model_buffer_size)
            with self.model.trace(self.boards[idx:slice_end]):
                residual_streams = self.model.residual_stream(self.residual_layer).output.save()

            def activation_scores_for_residual_stream(residual_stream):
                feature_activations = self.sparse_auto_encoder.encode(residual_stream)

                # Right now just get the value of the maximally activating token for each feature
                return torch.max(feature_activations, dim=0).values

            batched_scores_function = torch.vmap(activation_scores_for_residual_stream, in_dims=0)

            batched_scores = batched_scores_function(residual_streams.value)
            self.boards_to_activation_scores_map[idx:slice_end] = batched_scores

            idx += self.model_buffer_size
            torch.cuda.empty_cache()
            residual_streams.detach()

        self.activation_scores_initialized = True

    def get_specialized_features_indices(self):
        if not self.activation_scores_initialized:
            return []
        return "TODO Implement this"

    def visualize_feature(self, feature_idx):
        board_activations = self.boards_to_activation_scores_map[:,feature_idx].detach().numpy()
        top_board_indices = np.argsort(board_activations)[-self.relevant_boards_for_grouping:]
        top_boards = [self.boards[idx] for idx in top_board_indices]

        with self.model.trace(top_boards):
            residual_streams = self.model.residual_stream(self.residual_layer).output.save()

        pre_defined_max = self.boards_to_activation_scores_map[top_board_indices[0]][feature_idx]
        print(pre_defined_max)
        top_boards_and_heatmaps = []
        for i in range(len(top_boards)):
            board_and_heatmap_dict = {
                "board": top_boards[i],
                "heatmap": self.sparse_auto_encoder.encode(residual_streams[i])[:,feature_idx].detach(),
                "pre_defined_max": pre_defined_max
            }
            top_boards_and_heatmaps.append(board_and_heatmap_dict)

        return top_boards_and_heatmaps
