from flask import Flask, request, jsonify, Response
import subprocess

app = Flask(__name__)

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
            yield f"data: {line.decode('utf-8')}\n\n"  # Envia logs al cliente en formato SSE
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
            yield f"data: {line.decode('utf-8')}\n\n"  # Envia logs al cliente en formato SSE
        process.stdout.close()
        process.wait()

    try:
        return Response(generate_logs(), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)