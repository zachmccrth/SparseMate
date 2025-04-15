from flask import Blueprint, request, jsonify, render_template
from ..services.heatmap_service import generate_heatmap_from_feature
from ..utils.db_utils import get_features, get_tables

heatmap_bp = Blueprint('heatmap', __name__)


@heatmap_bp.route('/heatmap', methods=['POST', 'GET'])
def heatmap():
    if request.method == 'POST':
        # Check if the request is JSON
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415
        return answer_heatmap_request()
    else:

        return render_template("index.html", tables=get_tables())

@heatmap_bp.route('/features', methods=['POST'])
def fetch_features():
    table_name = request.json['table_name']
    features = get_features(table_name)
    return jsonify({'features': features})


def answer_heatmap_request():
    try:
        # Get the JSON data from the request
        data = request.json

        # Validate that required keys are present in the JSON payload
        table_name = data.get('table_name')
        feature_id = data.get('feature_id')

        if not table_name or not feature_id:
            return jsonify({"error": "Missing required parameters: 'table_name' and 'feature_id'"}), 400

        # Generate heatmaps for the given table and feature ID
        heatmaps = generate_heatmap_from_feature(table_name, feature_id)

        if not heatmaps:
            return jsonify({"error": "Failed to generate heatmap"}), 500

        return jsonify({"heatmaps": heatmaps})  # Return generated heatmaps

    except Exception as e:
        # Catch unexpected errors and return a generic error response
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
