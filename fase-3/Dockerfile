FROM my_model:latest

WORKDIR /app

# Instala Flask para la API REST
RUN pip install flask

# Copia los nuevos scripts y el archivo 'train.py'
COPY apirest.py /app/
COPY client.py /app/
COPY scripts/train.py /app/scripts/
COPY scripts/predict.py /app/scripts/

# Expone el puerto de la API
EXPOSE 5000

# Comando para iniciar la API
CMD ["python", "apirest.py"]
