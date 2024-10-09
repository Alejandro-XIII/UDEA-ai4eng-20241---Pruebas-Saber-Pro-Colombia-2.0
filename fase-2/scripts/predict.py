# fase-2/predict.py

import pandas as pd
import joblib
import numpy as np

# Cargar modelo
def load_model(model_path):
    return joblib.load(model_path)

# Hacer predicciones
def predict(model, input_data):
    preds = model.predict(input_data)
    preds = np.round(preds).astype(int)
    
    # Mapear a etiquetas
    mapping = {1: "bajo", 2: "medio-bajo", 3: "medio-alto", 4: "alto"}
    preds = np.vectorize(mapping.get)(preds)
    return preds

# Cargar datos de entrada
def load_input_data(input_path):
    data = pd.read_csv(input_path)
    return data.values[:, 1:]

if __name__ == "__main__":
    # Ruta del modelo y de los datos de entrada
    model_path = "/app/models/model.pkl"  # Ajusta la ruta según tu contenedor
    input_data_path = "/app/data/test.csv"
    
    # Cargar modelo y datos de entrada
    model = load_model(model_path)
    input_data = load_input_data(input_data_path)
    
    # Hacer predicciones
    predictions = predict(model, input_data)
    
    # Crear archivo de salida
    output = pd.DataFrame({"ID": pd.read_csv(input_data_path)["ID"], "RENDIMIENTO_GLOBAL": predictions})
    output.to_csv("/app/data/predictions.csv", index=False)
    print("Predicciones guardadas en /app/data/predictions.csv")
