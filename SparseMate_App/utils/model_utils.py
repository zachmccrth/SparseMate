import os
import re

import numpy as np
import torch
from dictionary_learning.dictionary import GOGS

# Global dictionary for caching models
model_cache = {}
projections_cache = {}


def table_name_to_filepath(table_name):
    """
    Convert a sanitized table name into its original file path.
    """
    match = re.match(r'RUN_(\d{4})(\d{4})(\w+)', table_name)
    if not match:
        return None

    month_day, time, trainer_type = match.groups()
    formatted_name = f"{month_day[:2]}{month_day[2:]}_{time[:2]}:{time[2:]}_{trainer_type}"
    base_dir = "/home/zachary/PycharmProjects/SparseMate/SAE_Models"
    filepath = os.path.join(base_dir, formatted_name)

    filepath = os.path.join(filepath,'ae.pt')

    return filepath


def load_model(table_name):
    """
    Load a model by its table name (cached globally).
    """
    if table_name in model_cache:
        return model_cache[table_name]

    filepath = table_name_to_filepath(table_name)
    if not filepath or not os.path.exists(filepath):
        print(f"Model file not found: {filepath}")
        return None


    try:
        model = GOGS.from_pretrained(filepath, device="cuda")
        model_cache[table_name] = model
        return model
    except Exception as e:
        print(f"Error loading model for {table_name}: {e}")
        return None

def projections_cache_access(table_name):
    """
    Lazy loading cache for projections (internal dot products actually)
    """
    if table_name not in projections_cache:
        model = load_model(table_name)
        if not model:
            return None
        projections = torch.matmul(model.basis_set, model.basis_set.T)
        projections_cache[table_name] = projections.cpu().detach().numpy()
    return projections_cache[table_name]


def get_similar_features(table_name, feature_id):
    """
    Get top N similar features using projection cache.
    """
    projections = projections_cache_access(table_name)

    try:
        feature_id = int(feature_id)
        projections = projections[feature_id]
        top_indices = np.argsort(projections)[-5:][::-1]
        top_values = projections[top_indices]
        return [{'feature': f'{int(idx)}', 'score': float(val)} for idx, val in zip(top_indices, top_values)]
    except KeyError:
        return None


def fetch_graph_data(table_name, feature_id, threshold=0.5, max_neighbors=10):
    """
    Fetch graph data for the given table and feature_id.

    Arguments:
    table_name -- The name of the table (used for the projection matrix).
    feature_id -- The ID of the selected feature.
    threshold -- Minimum similarity score for linking nodes.
    max_neighbors -- Maximum number of neighbors to fetch.

    Returns:
    A dictionary with `nodes` and `edges`.
    """

    projections = projections_cache_access(table_name)

    if feature_id < 0 or feature_id >= projections.shape[0]:
        return None

    # Node List: Include the main feature and the nearest neighbors
    nodes = [{'id': int(feature_id), 'label': f'f{feature_id}'}]
    edges = []

    # Find top neighbors above threshold similarity
    similarity_scores = projections[feature_id]
    neighbor_indices = np.argsort(similarity_scores)[::-1][1:max_neighbors + 1]  # Skip self (top index)

    for neighbor in neighbor_indices:
        score = similarity_scores[neighbor]
        if score >= threshold:
            nodes.append({'id': int(neighbor), 'label': f'f{neighbor}'})
            edges.append({'from': int(feature_id), 'to': int(neighbor), 'label': f"{score:.2f}"})

    return {'nodes': nodes, 'edges': edges}
