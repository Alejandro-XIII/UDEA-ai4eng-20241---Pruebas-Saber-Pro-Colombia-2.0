# Usar una imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo requirements.txt
COPY scripts/requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el directorio data y scripts al contenedor
COPY data ./data
COPY scripts ./scripts

# Establecer el comando por defecto para iniciar una shell interactiva
CMD ["bash"]