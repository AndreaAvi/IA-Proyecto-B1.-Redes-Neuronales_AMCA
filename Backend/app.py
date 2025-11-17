"""
Servidor Backend (Flask + SocketIO)
Expone la API para predecir, entrenar y obtener datos de la gráfica.
"""
import os
import threading
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS # Necesario para desarrollo local

# Importar toda la lógica del MLP
from mlp_logic import (
    MLP,
    X, y, Xn,
    mu as default_mu,
    sigma as default_sigma,
    train_model,
    load_norm_stats,
    save_norm_stats,
    get_plot_data
)

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo.npz")
STATS_PATH = os.path.join(BASE_DIR, "norm_stats.json")
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='/')
CORS(app) # Permitir peticiones desde el frontend (http://127.0.0.1:5000)
socketio = SocketIO(app, cors_allowed_origins="*")

# Modelo y stats en memoria
current_model = MLP()
current_mu = default_mu
current_sigma = default_sigma

try:
    current_model.load(MODEL_PATH)
    current_mu, current_sigma = load_norm_stats(STATS_PATH)
    print("Modelo y stats por defecto cargados en memoria.")
except FileNotFoundError:
    print("Advertencia: No se encontraron 'modelo.npz' o 'norm_stats.json'.")
    print("Por favor, ejecuta 'python backend/mlp_logic.py' primero.")
except Exception as e:
    print(f"Error cargando modelo/stats: {e}")

# -----------------------------
# RUTAS ESTÁTICAS (Sirven el Frontend)
# -----------------------------
@app.route('/')
def index():
    # Sirve el index.html principal
    return app.send_static_file('index.html')

# -----------------------------
# API: PREDICCIÓN
# -----------------------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        x = float(data['x'])
        y = float(data['y'])
        z = float(data['z'])

        # Normalizar
        xp = np.array([[x],[y],[z]])
        xp_n = (xp - current_mu) / current_sigma
        
        # Predecir
        prob = float(current_model.predict_prob(xp_n).item())
        clase = int(prob >= 0.5)

        return jsonify({"clase": clase, "probabilidad": prob})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------
# API: DATOS DE LA GRÁFICA
# -----------------------------
@app.route('/api/plot_data', methods=['GET'])
def api_plot_data():
    try:
        plot_data = get_plot_data(current_model, current_mu, current_sigma)
        return jsonify(plot_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# API: DESCARGAR MODELO
# -----------------------------
@app.route('/api/download_model', methods=['GET'])
def api_download_model():
    try:
        # Envía el archivo .npz
        return send_from_directory(BASE_DIR, 'modelo.npz', as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Modelo no encontrado"}), 404

# -----------------------------
# WEBSOCKETS: ENTRENAMIENTO
# -----------------------------
@socketio.on('connect')
def handle_connect():
    print('Cliente conectado al WebSocket')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

@socketio.on('start_training')
def handle_start_training():
    """ Inicia el entrenamiento en un hilo separado para no bloquear el servidor. """
    
    def socket_progress_callback(ep, total, loss, losses):
        """ Callback que envía progreso al cliente via WebSocket. """
        pct = int((ep / max(1, total)) * 100)
        # Usamos socketio.emit para enviar al cliente
        socketio.emit('training_progress', {
            'epoch': ep,
            'total_epochs': total,
            'loss': loss,
            'pct': pct,
            'losses': losses # Enviamos el historial de losses
        })

    def run_training_thread():
        """ El trabajo real de entrenamiento. """
        global current_model, current_mu, current_sigma
        print("Iniciando hilo de entrenamiento...")
        
        try:
            # Reseteamos el modelo y stats a los de los datos
            new_model = MLP()
            new_mu = X.mean(axis=1, keepdims=True)
            new_sigma = X.std(axis=1, keepdims=True) + 1e-8
            
            # Entrenar
            trained_model, final_losses = train_model(
                new_model, Xn, y, 
                epochs=8000, lr=0.05,
                progress_callback=socket_progress_callback
            )
            
            # Guardar y actualizar en memoria
            trained_model.save(MODEL_PATH)
            save_norm_stats(new_mu, new_sigma, STATS_PATH)
            
            current_model = trained_model
            current_mu = new_mu
            current_sigma = new_sigma
            
            print("Entrenamiento finalizado y modelo guardado.")
            socketio.emit('training_finished', {"status": "OK", "final_loss": final_losses[-1]})
        
        except Exception as e:
            print(f"Error en el hilo de entrenamiento: {e}")
            socketio.emit('training_error', {"error": str(e)})

    # Iniciar el hilo
    thread = threading.Thread(target=run_training_thread, daemon=True)
    thread.start()

# -----------------------------
# INICIAR SERVIDOR
# -----------------------------
if __name__ == '__main__':
    print(f"Sirviendo frontend desde: {FRONTEND_DIR}")
    print("Iniciando servidor Flask en http://127.0.0.1:5000")
    # Usamos socketio.run() en lugar de app.run()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)