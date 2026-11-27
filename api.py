# api.py - API de predicción con FastAPI (ruta /predict)
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Cargamos los pesos de la red 64 -> 32 -> 1 guardados al entrenar
M = np.load(Path(__file__).parent / "modelo.npz", allow_pickle=True)
columnas = list(M["columnas"])


def predecir_lote(df):
    # Mismo preprocesamiento del entrenamiento: dummies, orden de columnas y escalado
    x = pd.get_dummies(df, columns=["vehiculo"], dtype=float)
    x = x.reindex(columns=columnas, fill_value=0).to_numpy(dtype=float)
    x = (x - M["mean"]) / M["scale"]
    h = np.maximum(0, x @ M["W1"] + M["b1"])   # capa 1 (relu)
    h = np.maximum(0, h @ M["W2"] + M["b2"])   # capa 2 (relu)
    return (h @ M["W3"] + M["b3"]).ravel()     # salida: horas


app = FastAPI(title="API Logística")


class Pedido(BaseModel):
    distancia_km: float
    trafico: int
    clima: int
    peso_kg: float
    vehiculo: str


@app.get("/")
def inicio():
    return {"mensaje": "API lista. Usa POST /predict"}


@app.post("/predict")
def predict(p: Pedido):
    datos = p.model_dump() if hasattr(p, "model_dump") else p.dict()
    eta = predecir_lote(pd.DataFrame([datos]))[0]
    return {"tiempo_estimado": round(float(eta), 2)}
