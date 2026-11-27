# app.py - Dashboard logístico ML (Streamlit)
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from api import app as api_app, predecir_lote

st.set_page_config(page_title="Dashboard Logístico ML", layout="wide")

API_URL = "http://127.0.0.1:8000"


# ---------- La API corre dentro de la misma app (hilo en segundo plano) ----------
@st.cache_resource
def iniciar_api():
    try:
        import uvicorn
        servidor = uvicorn.Server(uvicorn.Config(
            api_app, host="127.0.0.1", port=8000, log_level="warning"))
        threading.Thread(target=servidor.run, daemon=True).start()
        for _ in range(20):
            try:
                requests.get(API_URL + "/", timeout=1)
                return True
            except Exception:
                time.sleep(0.5)
    except Exception:
        pass
    return False


API_ACTIVA = iniciar_api()


def estimar(datos):
    # Primero consulta la API (/predict); si no responde, usa el modelo directo
    if API_ACTIVA:
        try:
            r = requests.post(API_URL + "/predict", json=datos, timeout=5)
            return float(r.json()["tiempo_estimado"]), "API"
        except Exception:
            pass
    return float(predecir_lote(pd.DataFrame([datos]))[0]), "modelo directo"


# ---------- Datos: base SQLite pedidos_logisticos ----------
@st.cache_data
def cargar_pedidos():
    con = sqlite3.connect(Path(__file__).parent / "pedidos_logisticos.db")
    p = pd.read_sql("SELECT * FROM pedidos LIMIT 100", con)
    con.close()
    p["ETA"] = np.round(predecir_lote(p.drop(columns=["tiempo_entrega"])), 2)
    p["Pedido"] = [f"PED-{i:04d}" for i in range(1, len(p) + 1)]
    rng = np.random.default_rng(7)
    p["Fecha"] = pd.Timestamp("2026-11-16") + pd.to_timedelta(rng.integers(0, 7, len(p)), unit="D")
    umbral = round(float(np.percentile(p["ETA"], 90)), 1)  # el 10% más lento = riesgo alto
    p["Riesgo"] = np.where(p["ETA"] > umbral, "Alto", "Bajo")
    p["Estado"] = np.where(p["Riesgo"] == "Alto", "Retrasado", "En ruta")
    return p, umbral


def mostrar_grafico(fig):
    # Compatible con versiones nuevas y antiguas de Streamlit
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


pedidos, umbral = cargar_pedidos()
altos = pedidos[pedidos["Riesgo"] == "Alto"]

# ---------- KPIs ----------
st.title("Dashboard Logístico ML")
k1, k2, k3 = st.columns(3)
k1.metric("Pedidos Hoy", len(pedidos))
k2.metric("ETA Promedio", f"{pedidos['ETA'].mean():.2f} h")
k3.metric("Riesgo", f"{len(altos)} altos")
st.caption(f"Riesgo alto: ETA mayor a {umbral} horas")

# ---------- Formulario ----------
st.subheader("Formulario de predicción")
with st.form("formulario"):
    a, b, c = st.columns(3)
    distancia = a.number_input("Distancia (km)", 1.0, 3000.0, 180.0, 10.0)
    trafico = b.slider("Tráfico (1 = libre, 10 = muy alto)", 1, 10, 5)
    clima = c.selectbox("Clima", ["Bueno", "Malo"])
    d, e = st.columns(2)
    peso = d.number_input("Peso (kg)", 1.0, 1000.0, 100.0, 5.0)
    vehiculo = e.selectbox("Vehículo", ["moto", "auto", "camion"])
    enviar = st.form_submit_button("PREDECIR")

if enviar:
    eta, via = estimar({
        "distancia_km": distancia, "trafico": trafico,
        "clima": 0 if clima == "Bueno" else 1,
        "peso_kg": peso, "vehiculo": vehiculo})
    st.success(f"Resultado ETA: {eta:.2f} horas")
    st.caption(f"Riesgo: {'Alto' if eta > umbral else 'Bajo'} · consultado vía {via}")

# ---------- Tabla ----------
st.subheader("Tabla de pedidos")
st.dataframe(pedidos[["Pedido", "ETA", "Riesgo", "Estado"]], hide_index=True)

# ---------- Gráficos interactivos ----------
st.subheader("Gráficos")
g1, g2, g3 = st.columns(3)

with g1:
    eta_veh = pedidos.groupby("vehiculo", as_index=False)["ETA"].mean()
    mostrar_grafico(px.bar(eta_veh, x="vehiculo", y="ETA", color="vehiculo",
                           title="ETA por vehículo", height=300
                           ).update_layout(showlegend=False))
with g2:
    ret = altos.groupby("vehiculo", as_index=False).size().rename(columns={"size": "retrasadas"})
    mostrar_grafico(px.bar(ret, x="vehiculo", y="retrasadas", color="vehiculo",
                           title="Entregas retrasadas", height=300
                           ).update_layout(showlegend=False))
with g3:
    tend = pedidos.groupby("Fecha", as_index=False)["ETA"].mean()
    mostrar_grafico(px.line(tend, x="Fecha", y="ETA", markers=True,
                            title="Tendencia semanal", height=300))
