import torch
from flask import Flask, render_template, request, jsonify
import sqlite3
import numpy as np
import chess
import re

from dictionary_learning.dictionary import GOGS
from leela_interp.core.iceberg_board import IcebergBoard
from leela_interp.core.leela_board import LeelaBoard
import os

app = Flask(__name__)
# Global dictionary to cache models
model_cache = {}

def table_name_to_filepath(table_name):
    """
    Convert a sanitized table name back to the original file path.
    Example: 'RUN_04121624GOGSTrainer' -> '/home/zachary/PycharmProjects/SparseMate/SAE_Models/0413_16:45_GOGSTrainer'
    """
    # Extract date and model type from the table name
    match = re.match(r'RUN_(\d{4})(\d{4})(\w+)', table_name)
    if not match:
        return None

    month_day, time, trainer_type = match.groups()

    # Format back to original style with colons in the time
    formatted_name = f"{month_day[:2]}{month_day[2:]}_{time[:2]}:{time[2:]}_" \
                     f"{trainer_type}"

    # Construct the full file path
    base_dir = "/home/zachary/PycharmProjects/SparseMate/SAE_Models"
    filepath = os.path.join(base_dir, formatted_name)

    return filepath

# Function to load model based on table_name
def load_model(table_name):
    """
    Load model based on table_name and cache it in memory.
    """
    if table_name in model_cache:
        return model_cache[table_name]

    # Get original file path from table name
    filepath = table_name_to_filepath(table_name)
    if not filepath:
        print(f"Could not determine file path for table {table_name}")
        return None

    if not os.path.exists(filepath):
        print(f"Model file not found: {filepath}")
        return None

    try:

        model = GOGS.from_pretrained(filepath, device = "cuda")

        # Cache the model
        model_cache[table_name] = model
        print(f"Successfully loaded model for {table_name} from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model for {table_name}: {e}")
        return None


# Initialize models at application startup
@app.before_first_request
def initialize_models():
    """
    Load all models for available tables at application startup.
    """
    tables = get_tables()
    for table_name in tables:
        load_model(table_name)
    print(f"Initialized {len(model_cache)} models in cache")


# Database connection function
def get_db_connection():
    conn = sqlite3.connect('/home/zachary/PycharmProjects/SparseMate/SparseMate.sqlite')
    conn.row_factory = sqlite3.Row
    return conn


# Fetch all available tables
def get_tables():
    conn = get_db_connection()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return [table['name'] for table in tables]


# Fetch all distinct features from a given table
def get_features(table_name):
    conn = get_db_connection()
    query = f"SELECT DISTINCT feature FROM {table_name} ORDER BY feature"
    features = conn.execute(query).fetchall()
    conn.close()
    return [row['feature'] for row in features]


@app.route('/')
def index():
    tables = get_tables()
    first_table = tables[0] if tables else None
    features = get_features(first_table) if first_table else []
    return render_template('index.html', tables=tables, features=features)


@app.route('/features', methods=['POST'])
def fetch_features():
    table_name = request.json['table_name']
    features = get_features(table_name)
    return jsonify({'features': features})


@app.route('/heatmap', methods=['POST'])
def heatmap():
    data = request.json
    table_name = data['table_name']
    feature_id = data['feature_id']

    # Get the cached model for this table
    model = load_model(table_name)

    # Rest of your existing code...

    # Activated Features
    conn = get_db_connection()
    query = f"""
        SELECT fen, sq, value  
        FROM {table_name} 
        WHERE feature = ?
        AND value > 0.1 
        ORDER BY value DESC
    """
    rows = conn.execute(query, (feature_id,)).fetchall()

    # Control Boards
    control_query = f"""
        SELECT distinct(fen)
        FROM {table_name}
        LIMIT 10
    """
    control_rows = conn.execute(control_query, ()).fetchall()

    conn.close()

    boards = {}
    fens_ordered = []

    for row in rows:
        fen = row['fen']
        square = row['sq']
        value = row['value']

        leela_board = LeelaBoard.from_fen(fen)

        if fen not in boards:
            boards[fen] = np.zeros(64)
            fens_ordered.append(fen)

        idx = leela_board.sq2idx(square)
        boards[fen][idx] = value

    heatmap_images = []
    max_value = max(boards[fens_ordered[0]])

    control_boards = []
    for row in control_rows:
        fen = row['fen']

        if fen not in boards:
            boards[fen] = np.zeros(64)
        control_boards.append(fen)

    all_boards = fens_ordered[:50]
    all_boards.extend(control_boards)
    for fen in all_boards:
        board = chess.Board(fen)
        heatmap_data = boards[fen]
        iceberg_board = IcebergBoard(board=board, heatmap=heatmap_data, pre_defined_max=max_value)
        image_base64 = iceberg_board.render_to_base64()

        heatmap_images.append({'fen': fen, 'image': image_base64})

    return jsonify({'heatmaps': heatmap_images})

@app.route('/similar_features', methods=['POST'])
def similar_features():
    data = request.json
    table_name = data['table_name']
    feature_id = data['feature_id']

    # Get the cached model
    model = load_model(table_name)
    if model is None:
        return jsonify({'error': 'Model not available'}), 404

    # Use the model for your similarity calculations
    # ...

    projections = torch.matmul(model.basis_set, model.basis_set.T)



    # Example (replace with your actual implementation):
    similar = [
        {'feature': 'f123', 'score': 0.92},
        {'feature': 'f234', 'score': 0.89},
        {'feature': 'f345', 'score': 0.85}
    ]

    return jsonify({'similar_features': similar})


if __name__ == '__main__':
    app.run(debug=True)
