import requests

# URL base de la API
BASE_URL = "http://localhost:5000"

# Llamada al endpoint de entrenamiento
def call_train():
    response = requests.post(f"{BASE_URL}/train")
    print("Train response:", response.json())

# Llamada al endpoint de predicción
def call_predict(new_data):
    response = requests.post(f"{BASE_URL}/predict", json=new_data)
    print("Prediction response:", response.json())

if __name__ == "__main__":
    # Ejemplo de datos para predicción
    sample_data = {"feature1": 5.1, "feature2": 3.5, "feature3": 1.4}
    call_train()       # Ejecuta el entrenamiento
    call_predict(sample_data)  # Realiza una predicción