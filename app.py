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

st.set_page_config(page_title="Dashboard Logístico ML", page_icon="🚚", layout="wide")

API_URL = "http://127.0.0.1:8000"

VIOLETA = "#7C3AED"
TURQUESA = "#14B8A6"
ROSA = "#EC4899"
COLORES = {"moto": ROSA, "auto": TURQUESA, "camion": VIOLETA}

st.markdown("""
<style>
.encabezado {
    background: linear-gradient(120deg, #5B21B6 0%, #7C3AED 50%, #C026D3 100%);
    padding: 28px 34px; border-radius: 20px; margin-bottom: 22px;
    box-shadow: 0 14px 34px rgba(124, 58, 237, 0.35);
}
.encabezado h1 { color: #FFFFFF; margin: 0; font-size: 2.1rem; padding: 0; }
.encabezado p { color: #EADCFF; margin: 6px 0 0 0; }

div[data-testid="stMetric"] {
    background: rgba(124, 58, 237, 0.10);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-left: 6px solid #14B8A6;
    padding: 16px 20px; border-radius: 16px;
    box-shadow: 0 8px 22px rgba(124, 58, 237, 0.18);
    transition: transform .2s ease, box-shadow .2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 30px rgba(124, 58, 237, 0.30);
}
div[data-testid="stMetricValue"] { color: #8B5CF6 !important; font-weight: 800; }

div[data-testid="stForm"] {
    border: 1px solid rgba(124, 58, 237, 0.25); border-radius: 18px;
    background: rgba(124, 58, 237, 0.05);
    box-shadow: 0 8px 22px rgba(124, 58, 237, 0.14); padding: 22px;
}
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #7C3AED, #C026D3); color: #FFFFFF;
    border: 0; border-radius: 12px; font-weight: 700; padding: 8px 32px;
    box-shadow: 0 8px 18px rgba(124, 58, 237, 0.40);
    transition: transform .15s ease, box-shadow .15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px); color: #FFFFFF;
    box-shadow: 0 12px 24px rgba(124, 58, 237, 0.55);
}

div[data-testid="stDataFrame"] {
    border-radius: 16px; overflow: hidden;
    box-shadow: 0 8px 22px rgba(124, 58, 237, 0.16);
    border: 1px solid rgba(124, 58, 237, 0.20);
}
div[data-testid="stPlotlyChart"] {
    border-radius: 18px; padding: 8px;
    background: rgba(124, 58, 237, 0.06);
    border: 1px solid rgba(124, 58, 237, 0.18);
    box-shadow: 0 8px 22px rgba(124, 58, 237, 0.16);
}
div[data-testid="stHeading"] h3 {
    border-left: 6px solid #7C3AED; padding-left: 12px;
}
div[data-testid="stAlert"] { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)


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
    if API_ACTIVA:
        try:
            r = requests.post(API_URL + "/predict", json=datos, timeout=5)
            return float(r.json()["tiempo_estimado"]), "API"
        except Exception:
            pass
    return float(predecir_lote(pd.DataFrame([datos]))[0]), "modelo directo"


@st.cache_data
def cargar_pedidos():
    con = sqlite3.connect(Path(__file__).parent / "pedidos_logisticos.db")
    p = pd.read_sql("SELECT * FROM pedidos LIMIT 100", con)
    con.close()
    p["ETA"] = np.round(predecir_lote(p.drop(columns=["tiempo_entrega"])), 2)
    p["Pedido"] = [f"PED-{i:04d}" for i in range(1, len(p) + 1)]
    rng = np.random.default_rng(7)
    p["Fecha"] = pd.Timestamp("2026-11-16") + pd.to_timedelta(rng.integers(0, 7, len(p)), unit="D")
    umbral = round(float(np.percentile(p["ETA"], 90)), 1)
    p["Riesgo"] = np.where(p["ETA"] > umbral, "Alto", "Bajo")
    p["Estado"] = np.where(p["Riesgo"] == "Alto", "Retrasado", "En ruta")
    return p, umbral


def mostrar_grafico(fig):
    fig.update_layout(height=330, title_font_size=16,
                      margin=dict(l=10, r=10, t=55, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


pedidos, umbral = cargar_pedidos()
altos = pedidos[pedidos["Riesgo"] == "Alto"]

st.markdown(
    '<div class="encabezado"><h1>🚚 Dashboard Logístico ML</h1>'
    '<p>Predicción del tiempo de entrega con red neuronal</p></div>',
    unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
k1.metric("Pedidos Hoy", len(pedidos))
k2.metric("ETA Promedio", f"{pedidos['ETA'].mean():.2f} h")
k3.metric("Riesgo", f"{len(altos)} altos")
st.caption(f"Riesgo alto: ETA mayor a {umbral} horas")

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

st.subheader("Tabla de pedidos")
st.dataframe(pedidos[["Pedido", "ETA", "Riesgo", "Estado"]], hide_index=True)

st.subheader("Gráficos")
g1, g2, g3 = st.columns(3)

with g1:
    eta_veh = pedidos.groupby("vehiculo", as_index=False)["ETA"].mean().round(1)
    fig = px.bar(eta_veh, x="vehiculo", y="ETA", color="vehiculo", text="ETA",
                 color_discrete_map=COLORES, title="ETA promedio por vehículo (h)")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    mostrar_grafico(fig)

with g2:
    ret = altos.groupby("vehiculo", as_index=False).size().rename(columns={"size": "retrasadas"})
    fig = px.pie(ret, names="vehiculo", values="retrasadas", hole=0.55,
                 color="vehiculo", color_discrete_map=COLORES,
                 title="Entregas retrasadas por vehículo")
    fig.update_traces(textinfo="value+percent")
    fig.update_layout(legend=dict(orientation="h", y=-0.05))
    mostrar_grafico(fig)

with g3:
    tend = pedidos.groupby("Fecha", as_index=False)["ETA"].mean()
    tend["ETA"] = tend["ETA"].round(2)
    fig = px.line(tend, x="Fecha", y="ETA", markers=True, title="Tendencia semanal del ETA (h)")
    fig.update_traces(line_color=TURQUESA, line_width=3, marker_size=9,
                      fill="tozeroy", fillcolor="rgba(20,184,166,0.18)")
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickformat="%d %b")
    mostrar_grafico(fig)
