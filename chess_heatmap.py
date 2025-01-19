import chess
from flask import Flask, render_template, request, jsonify
import sqlite3
import numpy as np


from leela_interp.core.iceberg_board import IcebergBoard
from leela_interp.core.leela_board import LeelaBoard

# Initialize Flask app
app = Flask(__name__)

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('/home/zachary/PycharmProjects/SparseMate/data_tools/layer_6.db')
    conn.row_factory = sqlite3.Row
    return conn

# Route to serve the main page
@app.route('/')
def index():
    # Fetch feature IDs to display as options
    conn = get_db_connection()
    features = conn.execute('SELECT DISTINCT feature FROM activations ORDER BY feature').fetchall()
    features = [row['feature'] for row in features]
    conn.close()
    return render_template('index.html', features=features)

# Route to generate heatmap for a selected feature
@app.route('/heatmap', methods=['POST'])
def heatmap():
    feature_id = request.json['feature_id']

    # Query database for activation values
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT fen, sq, value 
        FROM activations 
        WHERE feature = ?
        ORDER BY value DESC
    ''', (feature_id,)).fetchall()
    conn.close()

    # Process data to create a heatmap per board
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


    # Generate heatmaps
    heatmap_images = []
    idx = 0

    for fen in fens_ordered:

        board: chess.Board = chess.Board(fen)

        heatmap_data = boards[fen]
        # heatmap_data = np.linspace(0, 4, num=64)
        iceberg_board = IcebergBoard(board=board, heatmap=heatmap_data)

        image_base64 = iceberg_board.render_to_base64()

        heatmap_images.append(image_base64)

        idx += 1
        if idx > 50:
            break


    return jsonify({'heatmaps': heatmap_images})

if __name__ == '__main__':
    app.run(debug=True)
