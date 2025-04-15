import numpy as np
import chess
from leela_interp.core.iceberg_board import IcebergBoard
from leela_interp.core.lc0 import Lc0Model
from leela_interp.core.leela_board import LeelaBoard
from ..utils.db_utils import get_db_connection
from ..utils.model_utils import load_model



def generate_heatmap_from_feature(table_name, feature_id):
    """
    Generates heatmap data for a given table name and feature ID.
    """
    conn = get_db_connection()
    query = f"""
        SELECT fen, sq, value  
        FROM {table_name} 
        WHERE feature = ?
        AND value > 0.1 
        ORDER BY value DESC
    """
    rows = conn.execute(query, (feature_id,)).fetchall()
    conn.close()

    # Create heatmaps for features
    boards = {}
    fens_ordered = []
    for row in rows:
        fen, square, value = row['fen'], row['sq'], row['value']
        leela_board = LeelaBoard.from_fen(fen)
        idx = leela_board.sq2idx(square)
        boards[fen] = boards.get(fen, np.zeros(64))
        boards[fen][idx] = value
        fens_ordered.append(fen)

    # Render the heatmaps as Base64 images
    heatmap_images = []
    for fen in fens_ordered[:50]:
        board = chess.Board(fen)
        heatmap_data = boards.get(fen, np.zeros(64))
        iceberg_board = IcebergBoard(board=board, heatmap=heatmap_data)
        heatmap_images.append({'fen': fen, 'image': iceberg_board.render_to_base64()})

    return heatmap_images
