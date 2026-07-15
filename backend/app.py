"""
Main Flask Application
BloodPrint AI
"""

from flask import Flask, jsonify
from flask_cors import CORS

from routes.predict import predict_bp
from routes.history import history_bp
from routes.report import report_bp

from utils.database import create_database

app = Flask(__name__)

# Enable CORS
CORS(app)

# Create database
create_database()

# Register Blueprints
app.register_blueprint(predict_bp)
app.register_blueprint(history_bp)
app.register_blueprint(report_bp)


@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "project": "BloodPrint AI",

        "version": "1.0",

        "status": "Running",

        "developer": "Namrata Nagraj"

    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "Healthy",

        "backend": "Running"

    })


@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "message": "API endpoint not found."

    }), 404


@app.errorhandler(500)
def server_error(error):

    return jsonify({

        "success": False,

        "message": "Internal Server Error."

    }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
