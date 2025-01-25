# Flask App Update
from flask import Flask, render_template, request, jsonify
import sqlite3
import numpy as np
import chess
from leela_interp.core.iceberg_board import IcebergBoard
from leela_interp.core.leela_board import LeelaBoard

app = Flask(__name__)

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('/home/zachary/PycharmProjects/SparseMate/data_tools/layer_6_attentionseeking_smalld.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_puzzles_db():
    conn = sqlite3.connect('identifier.sqlite')
    return conn

# Route to serve the main page
@app.route('/')
def index():
    conn = get_db_connection()
    features = conn.execute('SELECT DISTINCT feature FROM activations ORDER BY feature').fetchall()
    features = [row['feature'] for row in features]
    conn.close()
    return render_template('index.html', features=features)


@app.route('/heatmap', methods=['POST'])
def heatmap():
    feature_id = request.json['feature_id']
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT fen, sq, value 
        FROM activations 
        WHERE feature = ?
        ORDER BY value DESC
    ''', (feature_id,)).fetchall()
    conn.close()

    boards = {}
    fens_ordered = []

    puzzle_db = get_puzzles_db()

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

    for fen in fens_ordered[:50]:
        board = chess.Board(fen)
        heatmap_data = boards[fen]
        iceberg_board = IcebergBoard(board=board, heatmap=heatmap_data, pre_defined_max=max_value)
        image_base64 = iceberg_board.render_to_base64()

        if puzzle_db is not None:
            puzzle_info = puzzle_db.execute("""
            SELECT Moves, Themes FROM lichess_db_puzzle WHERE fen = ?""", (fen,)).fetchall()


        if len(puzzle_info) > 0:
            heatmap_images.append({'fen': fen, 'image': image_base64, 'moves': puzzle_info[0][0], 'themes': puzzle_info[0][1] })
        else:
            heatmap_images.append({'fen': fen, 'image': image_base64})
        puzzle_info = None


    puzzle_db.close()
    return jsonify({'heatmaps': heatmap_images})

if __name__ == '__main__':
    app.run(debug=True)
