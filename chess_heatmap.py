from flask import Flask, render_template, request, jsonify
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('/home/zachary/PycharmProjects/SparseMate/data_tools/activations.db')
    conn.row_factory = sqlite3.Row
    return conn

# Route to serve the main page
@app.route('/')
def index():
    # Fetch feature IDs to display as options
    conn = get_db_connection()
    features = conn.execute('SELECT DISTINCT feature FROM activations').fetchall()
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
    ''', (feature_id,)).fetchall()
    conn.close()

    # Process data to create a heatmap per board
    boards = {}
    for row in rows:
        fen = row['fen']
        square = row['sq']
        value = row['value']

        if fen not in boards:
            boards[fen] = np.zeros((8, 8))

        file = ord(square[0]) - ord('a')  # Convert 'a'-'h' to 0-7
        rank = 8 - int(square[1])         # Convert '1'-'8' to 7-0
        boards[fen][rank, file] = value

    # Generate heatmaps
    heatmap_images = []
    for fen, heatmap_data in boards.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Activation Value')
        plt.title(f'Feature: {feature_id} | FEN: {fen}')
        plt.axis('off')

        # Save plot to a string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.show()
        plt.close()

        heatmap_images.append(image_base64)

    return jsonify({'heatmaps': heatmap_images})

if __name__ == '__main__':
    app.run(debug=True)
