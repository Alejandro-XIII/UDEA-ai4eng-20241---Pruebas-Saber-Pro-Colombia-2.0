# fase-2/train.py

import pandas as pd
from xgboost import XGBRegressor
import joblib  # Para guardar el modelo

# Cargar datos de entrenamiento
def load_data(train_path):
    data = pd.read_csv(train_path)
    X = data.values[:, 1:-1]
    y = data["RENDIMIENTO_GLOBAL"].values
    return X, y

# Entrenar el modelo
def train_model(X, y):
    model = XGBRegressor()
    model.fit(X, y)
    return model

# Guardar el modelo entrenado
def save_model(model, model_path):
    joblib.dump(model, model_path)

if __name__ == "__main__":
    # Ruta de los datos de entrenamiento y donde se guardará el modelo
    train_data_path = "/app/data/train.csv"  # Ajusta la ruta según tu contenedor
    model_save_path = "/app/models/model.pkl"
    
    # Cargar datos
    X, y = load_data(train_data_path)
    
    # Entrenar y guardar el modelo
    model = train_model(X, y)
    save_model(model, model_save_path)
    print(f"Modelo guardado en {model_save_path}")
