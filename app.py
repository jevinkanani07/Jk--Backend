from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model, scaler and columns
model = pickle.load(open("Logistic_heart.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return "Backend Running Successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Build input vector in same order as columns
        input_data = []
        for col in columns:
            input_data.append(data[col])

        final_input = np.array(input_data).reshape(1, -1)
        scaled_data = scaler.transform(final_input)
        prediction = model.predict(scaled_data)[0]

        return jsonify({
            "prediction": int(prediction),
            "status": "Heart Disease" if prediction == 1 else "No Heart Disease"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
