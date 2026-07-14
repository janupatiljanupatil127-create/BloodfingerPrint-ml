from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

CORS(app)


@app.route("/")
def home():
    return {
        "project": "BloodPrint AI",
        "version": "1.0",
        "status": "Running"
    }


if __name__ == "__main__":
    app.run(debug=True)
