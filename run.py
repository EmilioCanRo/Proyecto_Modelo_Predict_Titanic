from flask import Flask, request
import os
from app.src.models.predict import predict_pipeline
import warnings

# Quitar warnings innecesarios de la salida
warnings.filterwarnings("ignore")

# -*- coding: utf-8 -*-
app = Flask(__name__)

# When running this app on the local machine, default the port to 8080
port = int(os.getenv("PORT", 8080))


# usando el decorador @app.route para gestionar los enrutadores (Método GET)
@app.route("/", methods=["GET"])
def root():
    """
    Función para gestionar la salida de la ruta raíz.

    Returns:
       dict.  Mensaje de salida
    """
    return {"Proyecto": "Mod. 4 - Ciclo de vida de modelos IA"}


# ruta para el lanzar el pipeline de inferencia (Método POST)
@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Función de lanzamiento del pipeline de inferencia.

    Returns:
       dict.  Mensaje de salida (predicción)
    """

    # Obtener los datos pasados por el request
    data = request.get_json()

    # Lanzar la ejecución del pipeline de inferencia
    y_pred = predict_pipeline(data)

    return {"Predicted value": y_pred}


# main
if __name__ == "__main__":
    # ejecución de la app
    app.run(host="0.0.0.0", port=port, debug=True)
