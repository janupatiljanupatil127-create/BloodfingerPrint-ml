from flask import Flask
from flask_cors import CORS

from routes.predict import predict_bp

app = Flask(__name__)

CORS(app)

app.register_blueprint(predict_bp)


@app.route("/")
def home():

    return {
        "project": "BloodPrint AI",
        "status": "Running",
        "version": "1.0"
    }


if __name__ == "__main__":

    app.run(debug=True)
