from flask import Flask

from SparseMate_App.routes.index_routes import index_bp
from .routes.heatmap_routes import heatmap_bp
from .routes.similarity_routes import similarity_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(index_bp)

app.register_blueprint(heatmap_bp)
app.register_blueprint(similarity_bp)



if __name__ == '__main__':
    app.run(debug=True)
