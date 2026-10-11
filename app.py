
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

saved = joblib.load("ckd_model.pkl")
model = saved["model"]
columns = saved["columns"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "CKD Prediction API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Provide a JSON object"}), 400

        row = pd.DataFrame([data])
        row = pd.get_dummies(row, dummy_na=True)
        row = row.reindex(columns=columns, fill_value=0)

        prediction = model.predict(row)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
