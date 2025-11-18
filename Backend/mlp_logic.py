"""
    Andrea Marcela Cáceres Avitia (Inteligencia Artificial 2026-I)
    IA. Proyecto B1: Redes Neuronales
    Clase: mlp_logic.py (perceptrón multicapa)
    Descripción: Lógica de red neuronal para clasificación binaria.
    Objetivo: Red neuronal que decide si un punto dado pertenece a la Clase 0
        (amarillo) o a la Clase 1 (rojo).
        Arquitectura: 3 -> 8 -> 1
    Elementos:
        Capas de entrada: 3 neuronas (x,y,z).
        Capa oculta: 8 neuronas.
        Capa de salifda: 1 neurona.
        Pesos:
            w1: Matriz de pesos 3x8.
            w2: Matriz de pesos 8x1.
        Sigmoide:
            sigmoid(x) = \frac{1}{1 + e^{-x}}
"""
# 
import os
import json
import numpy as np

np.random.seed(42)

# Implementa función de activación (sigmoide).
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calcula la derivada del sigmoide.
def sigmoid_deriv(s):
    return s * (1 - s)

#Implementa la función de costo.
def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Constructor de la red (3 a 8 a 1)
class MLP:
    def __init__(self, n_in=3, n_hidden=8, n_out=1):
        # Los sesgos inician en cero y los pesos son aleatorios.
        limit1 = np.sqrt(6 / (n_in + n_hidden))
        self.W1 = np.random.uniform(-limit1, limit1, (n_hidden, n_in))
        self.b1 = np.zeros((n_hidden, 1))

        # Inicialización de Xavier Glorot (la ini perfecta :)).
        # Inicializa los pesos en un rango perfecto.
        limit2 = np.sqrt(6 / (n_hidden + n_out))
        self.W2 = np.random.uniform(-limit2, limit2, (n_out, n_hidden))
        self.b2 = np.zeros((n_out, 1))

    # Propagación hacia adelante.
    def forward(self, x):
        z1 = self.W1.dot(x) + self.b1 # Aplica  (W * x) + b.
        a1 = sigmoid(z1) # Aplica el sigmoide.
        z2 = self.W2.dot(a1) + self.b2 # Aplica  (W * x) + b.
        a2 = sigmoid(z2) # Aplica el sigmoide.
        return a2, (x, z1, a1, z2, a2) # Devuelve la predicción y una memoria.

    # ❤️❤️❤️ PROPAGACIÓN HACIA ATRÁS ❤️❤️❤️
    def backward(self, cache, y_true):
        x, z1, a1, z2, a2 = cache
        m = x.shape[1] # m es el número de muestras.

        dz2 = a2 - y_true # Derivada del costo + sigmoide.
        dW2 = (1/m) * dz2.dot(a1.T) # Gradiente para W2.
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True) # Gradiente para b2.

        da1 = self.W2.T.dot(dz2) # Propaga el error a la capa anterior.
        dz1 = da1 * sigmoid_deriv(a1) # Aplicar derivada de sigmoide.
        dW1 = (1/m) * dz1.dot(x.T) # Gradiente para W1.
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True) # Gradiente para b1.

        return dW1, db1, dW2, db2 # Regresa los gradientes obtenidos.

# Actualiza los pesos y sesgos con los gradientes obtenidos.
    def update_params(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

# Corre el forward y saca la probabilidad.
    def predict_prob(self, x):
        a2, _ = self.forward(x)
        return a2

# Usa el umbral para discernir a dónde pertenece.
    def predict(self, x):
        return (self.predict_prob(x) >= 0.5).astype(int)

# Carga los pesos y sesgos a un archivo.
    def save(self, path="modelo.npz"):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path="modelo.npz"):
        d = np.load(path)
        self.W1 = d["W1"]; self.b1 = d["b1"]
        self.W2 = d["W2"]; self.b2 = d["b2"]

# Se meten las 6 muestras dadas :3.
muestras = [
    (3.0, 12.0, -5.0),
    (8.0, 12.0, -2.0),
    (3.0, 0.50, -1.0),
    (10.0, -0.50, -1.0),
    (5.0, 5.0, 2.0),
    (12.0, -0.50, 1.0)
]

# Se asignan las clases.
clases = {
    (3.0, 12.0, -5.0): 0,
    (8.0, 12.0, -2.0): 0,
    (3.0, 0.50, -1.0): 0,
    (10.0, -0.50, -1.0): 0,
    (5.0, 5.0, 2.0): 1,
    (12.0, -0.50, 1.0): 1
}

X = np.array(muestras).T
y = np.array([clases[p] for p in muestras]).reshape(1, -1) # Formato para que esté de 1 a 6.

# Mu es la media y sigma la desviación estándar.
# Xn = (X - mu) / sigma --- Estandarización.
# media de 0 y desviación de 1.
mu = X.mean(axis=1, keepdims=True)
sigma = X.std(axis=1, keepdims=True) + 1e-8
Xn = (X - mu) / sigma

#------------------------------------------------------------------------------------------
# ENTRENAMIENTO
# Todos los pasos se repiten 8000 veces.
def train_model(model, Xn, y, epochs=8000, lr=0.05, progress_callback=None):
    losses = []
    for ep in range(epochs):
        a2, cache = model.forward(Xn) # Paso 1: Forward.
        loss = float(np.mean(binary_cross_entropy(y, a2))) # Paso 2: Medir Error.
        losses.append(loss)

        grads = model.backward(cache, y) # Paso 3: Backward.
        model.update_params(grads, lr) # Paso 4: Actualizar.

        if progress_callback and (ep % max(1, epochs // 200) == 0):
            # Llama al callback para enviar progreso.
            progress_callback(ep, epochs, loss, losses)
    
    if progress_callback:
        progress_callback(epochs, epochs, losses[-1], losses)
    return model, losses

# Guardar datos con JSON.
def save_norm_stats(mu, sigma, path="norm_stats.json"):
    stats = {
        "mu": mu.flatten().tolist(),
        "sigma": sigma.flatten().tolist()
    }
    with open(path, "w") as f:
        json.dump(stats, f)

# Cargar datos con JSON.
def load_norm_stats(path="norm_stats.json"):
    with open(path, "rb") as f:
        stats = json.load(f)
    mu_l = np.array(stats["mu"]).reshape(-1, 1)
    sigma_l = np.array(stats["sigma"]).reshape(-1, 1)
    return mu_l, sigma_l

# Gráfica.
def get_plot_data(model, mu, sigma, samples=2000):
    """
    Genera los datos para la gráfica 3D y los devuelve como
    diccionario de listas.
    """
    X_full = np.array(muestras)
    mins = X_full.min(axis=0) - 1
    maxs = X_full.max(axis=0) + 1

    rand = np.random.uniform(mins, maxs, (samples, 3))
    rand_n = ((rand.T - mu) / sigma).T
    probs = model.predict_prob(rand_n.T).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    # Regiones de decisión
    region0 = rand[preds==0].tolist()
    region1 = rand[preds==1].tolist()

    # Datos originales.
    muestras0 = [p for p in muestras if clases[p] == 0]
    muestras1 = [p for p in muestras if clases[p] == 1]
    
    return {
        "region0": region0,
        "region1": region1,
        "muestras0": muestras0,
        "muestras1": muestras1
    }

# Preentrenar red. Sólo lo ejecuté una vez desde terminal para generar el modelo inicial.
if __name__ == "__main__":
    print("Iniciando pre-entrenamiento del modelo por defecto...")
    
    # Rutas relativas al script.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "modelo.npz")
    stats_path = os.path.join(base_dir, "norm_stats.json")

    # Entrenar.
    default_model = MLP()
    default_model, _ = train_model(default_model, Xn, y, epochs=8000, lr=0.05)
    
    # Guardar modelo y stats.
    default_model.save(model_path)
    save_norm_stats(mu, sigma, stats_path)
    
    print(f"Modelo por defecto guardado en: {model_path}")
    print(f"Stats por defecto guardados en: {stats_path}")