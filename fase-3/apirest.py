from flask import Flask, request, jsonify
import joblib
import pandas as pd
from scripts.train import train_model
from scripts.predict import predict

app = Flask(__name__)

# Endpoint de entrenamiento
@app.route('/train', methods=['POST'])
def train():
    train_model()  # Ejecuta el entrenamiento
    return jsonify({"status": "Model trained successfully"}), 200

# Endpoint de predicción
@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    df = pd.DataFrame([data])  # Convierte el JSON en DataFrame
    prediction = predict(df)  # Llama al script de predicción
    return jsonify({"prediction": prediction}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)