# Usar una imagen base de Python
FROM python:3.10

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY requirements.txt .
COPY scripts/train.py .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Crear un directorio para los datos
RUN mkdir data

# Definir el comando por defecto
ENTRYPOINT ["python", "train.py"]
