from flask import Flask, render_template, request, jsonify
import sqlite3
import numpy as np
import chess
from leela_interp.core.iceberg_board import IcebergBoard
from leela_interp.core.leela_board import LeelaBoard

app = Flask(__name__)


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

    # Activated Features
    conn = get_db_connection()
    query = f"""
        SELECT fen, sq, value  
        FROM {table_name} 
        WHERE feature = ? 
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


if __name__ == '__main__':
    app.run(debug=True)
