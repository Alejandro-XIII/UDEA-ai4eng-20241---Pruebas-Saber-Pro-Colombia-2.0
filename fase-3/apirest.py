from flask import Flask, jsonify, Response

import subprocess
import pandas as pd

app = Flask(__name__)

# Ruta para la página de inicio
@app.route('/')
def index():
    return """
    <h1>Bienvenido a la API</h1>
    <p>Utiliza los siguientes endpoints:</p>
    <ul>
        <li><b>/train</b>: Entrena el modelo</li>
        <li><b>/predict</b>: Realiza predicciones</li>
    </ul>
    <p>Visita <a href="/train">/train</a> para entrenar o <a href="/predict">/predict</a> para predecir.</p>
    """

# Endpoint para entrenar el modelo con logs en tiempo real
@app.route('/train', methods=['GET'])
def train():
    def generate_logs():
        # Ejecuta train_model desde train.py como un proceso externo
        process = subprocess.Popen(
            ["python", "-u", "scripts/train.py"],  # -u para logs sin buffering
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        for line in iter(process.stdout.readline, b""):  # Itera sobre cada línea de salida
            yield f"Proceso: {line.decode('utf-8')}\n\n"  # Envía logs al cliente en formato SSE
        process.stdout.close()
        process.wait()

    return Response(generate_logs(), mimetype='text/event-stream')

# Endpoint para realizar predicciones con logs en tiempo real
@app.route('/predict', methods=['GET'])
def make_prediction():
    def generate_logs():
        # Ejecuta predict desde predict.py como un proceso externo
        process = subprocess.Popen(
            ["python", "-u", "scripts/predict.py"],  # -u para logs sin buffering
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        for line in iter(process.stdout.readline, b""):  # Itera sobre cada línea de salida
            yield f"Proceso: {line.decode('utf-8')}\n\n"  # Envía logs al cliente en formato SSE
        process.stdout.close()
        process.wait()

        yield "Proceso: Cargando predicciones...\n\n"

        # Leer el archivo de predicciones y convertirlo a JSON
        df = pd.read_csv('/app/data/submission.csv')
        json_data = df.to_json(orient='records')  # Convierte el DataFrame a una lista de diccionarios en JSON
        yield f"data: {json_data}\n\n"

    try:
        return Response(generate_logs(), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
