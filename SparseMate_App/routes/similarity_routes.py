from flask import Blueprint, request, jsonify
from ..utils.db_utils import get_features, get_tables
from ..utils.model_utils import get_similar_features, fetch_graph_data

similarity_bp = Blueprint('model', __name__)




@similarity_bp.route('/similar_features', methods=['POST'])
def similar_features():
    data = request.json
    table_name = data['table_name']
    feature_id = data['feature_id']
    similar = get_similar_features(table_name, feature_id)
    if not similar:
        return jsonify({'error': 'Failed to fetch similar features'}), 500
    return jsonify({'similar_features': similar})


@similarity_bp.route('/graph_data', methods=['POST'])
def graph_data():
    data = request.json
    table_name = data.get('table_name')
    feature_id = data.get('feature_id')

    if not table_name or not isinstance(table_name, str):
        return jsonify({'error': 'Invalid table_name'}), 400

    if feature_id is None:
        return jsonify({'error': 'feature_id is required'}), 400

    try:
        feature_id = int(feature_id)
    except ValueError:
        return jsonify({'error': 'feature_id must be an integer'}), 400

    # Fetch the graph data
    graph_data = fetch_graph_data(table_name, feature_id)
    if not graph_data:
        return jsonify({'error': 'Failed to fetch graph data'}), 500

    return jsonify(graph_data)
