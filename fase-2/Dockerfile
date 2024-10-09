# Usa una imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY scripts/train.py /app/train.py
COPY scripts/predict.py /app/predict.py
COPY data/train.csv /app/data/train.csv
COPY requirements.txt /app/requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando por defecto al ejecutar el contenedor (ajusta según sea necesario)
CMD ["python", "train.py", "--data_file", "data/train.csv", "--model_file", "model.json", "--overwrite_model"]