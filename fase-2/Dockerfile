# fase-2/Dockerfile

# Usar una imagen base de Python
FROM python:3.8-slim

# Instalar las dependencias necesarias
RUN pip install pandas numpy scikit-learn xgboost joblib

# Crear directorios para el código y los datos
WORKDIR /app
COPY . /app

# Comando por defecto (puedes cambiar esto según si vas a entrenar o predecir)
CMD ["python", "train.py"]
