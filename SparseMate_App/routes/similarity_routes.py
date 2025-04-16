from flask import Blueprint, request, jsonify
from ..utils.db_utils import get_features, get_tables
from ..utils.model_utils import get_similar_features

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
