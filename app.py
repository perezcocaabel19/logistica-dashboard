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
.titulo h1 {
    font-size: 2.3rem; font-weight: 800; margin: 0; padding: 0;
    background: linear-gradient(90deg, #7C3AED, #C026D3, #EC4899);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.titulo p { opacity: .7; margin: 2px 0 18px 0; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(124,58,237,.20), rgba(236,72,153,.08));
    border-right: 1px solid rgba(124,58,237,.30);
}
.lateral { font-size: 1.25rem; font-weight: 800; margin-bottom: 4px; }
.lateral small { display: block; font-weight: 400; opacity: .7; font-size: .85rem; }

div[data-testid="stForm"] {
    border: 1px solid rgba(124,58,237,.30); border-radius: 18px;
    background: rgba(124,58,237,.07); padding: 18px;
    box-shadow: 0 8px 22px rgba(124,58,237,.18);
}
div[data-testid="stFormSubmitButton"] { width: 100%; }
div[data-testid="stFormSubmitButton"] button {
    width: 100%; background: linear-gradient(135deg, #7C3AED, #C026D3); color: #FFFFFF;
    border: 0; border-radius: 12px; font-weight: 700; padding: 10px 18px;
    box-shadow: 0 8px 18px rgba(124,58,237,.45);
    transition: transform .15s ease, box-shadow .15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px); color: #FFFFFF; box-shadow: 0 12px 24px rgba(124,58,237,.6);
}
[data-testid="stSlider"] [role="group"] { filter: hue-rotate(-96deg) saturate(1.15); }

.hero {
    background: linear-gradient(135deg, #5B21B6 0%, #7C3AED 55%, #C026D3 100%);
    border-radius: 22px; padding: 22px 26px; color: #FFFFFF; min-height: 132px;
    box-shadow: 0 14px 34px rgba(124,58,237,.40);
}
.hero .eti { font-size: .85rem; letter-spacing: .12em; text-transform: uppercase; opacity: .85; }
.hero .val { font-size: 2.6rem; font-weight: 800; line-height: 1.15; }
.hero .val small { font-size: 1.1rem; font-weight: 500; opacity: .9; }
.hero .sub { font-size: .9rem; opacity: .95; margin-top: 4px; }
.pill {
    display: inline-block; padding: 2px 12px; border-radius: 999px;
    font-weight: 700; font-size: .8rem; margin-right: 8px;
}
.pill.alto { background: #EC4899; color: #FFFFFF; }
.pill.bajo { background: #14B8A6; color: #FFFFFF; }

.kpi {
    display: flex; align-items: center; gap: 12px; min-height: 132px;
    padding: 16px 16px; border-radius: 22px;
    background: rgba(124,58,237,.10); border: 1px solid rgba(124,58,237,.25);
    box-shadow: 0 8px 22px rgba(124,58,237,.18);
    transition: transform .2s ease, box-shadow .2s ease;
}
.kpi:hover { transform: translateY(-4px); box-shadow: 0 14px 30px rgba(124,58,237,.32); }
.kpi .ico {
    font-size: 1.5rem; width: 46px; height: 46px; border-radius: 16px; flex: none;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(124,58,237,.30), rgba(20,184,166,.30));
}
.kpi .lab { font-size: .85rem; opacity: .75; }
.kpi .num { font-size: 1.5rem; font-weight: 800; color: #8B5CF6; line-height: 1.1; white-space: nowrap; }

.seccion {
    font-size: 1.15rem; font-weight: 700; margin: 26px 0 10px 0;
    display: flex; align-items: center; gap: 10px;
}
.seccion::before {
    content: ""; width: 34px; height: 6px; border-radius: 6px;
    background: linear-gradient(90deg, #7C3AED, #EC4899);
}

div[data-testid="stDataFrame"] {
    border-radius: 18px; overflow: hidden; border: 1px solid rgba(124,58,237,.22);
    box-shadow: 0 8px 22px rgba(124,58,237,.16);
}
div[data-testid="stPlotlyChart"] {
    border-radius: 20px; padding: 8px; background: rgba(124,58,237,.06);
    border: 1px solid rgba(124,58,237,.20); box-shadow: 0 8px 22px rgba(124,58,237,.16);
}
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


def color_riesgo(valor):
    if valor == "Alto":
        return "background-color: rgba(236,72,153,.28); font-weight: 700"
    return "background-color: rgba(20,184,166,.22)"


pedidos, umbral = cargar_pedidos()
altos = pedidos[pedidos["Riesgo"] == "Alto"]

if "resultado" not in st.session_state:
    st.session_state["resultado"] = None

with st.sidebar:
    st.markdown('<div class="lateral">🚚 Nuevo pedido<small>Ingresa los datos del envío</small></div>',
                unsafe_allow_html=True)
    with st.form("formulario"):
        distancia = st.number_input("Distancia (km)", 1.0, 3000.0, 180.0, 10.0)
        trafico = st.slider("Tráfico (1 = libre, 10 = muy alto)", 1, 10, 5)
        clima = st.selectbox("Clima", ["Bueno", "Malo"])
        peso = st.number_input("Peso (kg)", 1.0, 1000.0, 100.0, 5.0)
        vehiculo = st.selectbox("Vehículo", ["moto", "auto", "camion"])
        enviar = st.form_submit_button("PREDECIR")
    st.caption("● API activa" if API_ACTIVA else "● Modelo directo")

if enviar:
    eta, via = estimar({
        "distancia_km": distancia, "trafico": trafico,
        "clima": 0 if clima == "Bueno" else 1,
        "peso_kg": peso, "vehiculo": vehiculo})
    st.session_state["resultado"] = {"eta": eta, "via": via}

st.markdown('<div class="titulo"><h1>Dashboard Logístico ML</h1>'
            '<p>Predicción del tiempo de entrega con red neuronal</p></div>',
            unsafe_allow_html=True)

res = st.session_state["resultado"]
if res is None:
    hero = ('<div class="hero"><div class="eti">Predicción</div><div class="val">--</div>'
            '<div class="sub">Completa el formulario y pulsa PREDECIR</div></div>')
else:
    alto = res["eta"] > umbral
    hero = (f'<div class="hero"><div class="eti">Resultado ETA</div>'
            f'<div class="val">{res["eta"]:.2f} <small>horas</small></div>'
            f'<div class="sub"><span class="pill {"alto" if alto else "bajo"}">'
            f'Riesgo {"Alto" if alto else "Bajo"}</span>consultado vía {res["via"]}</div></div>')


def tarjeta(icono, etiqueta, valor):
    return (f'<div class="kpi"><div class="ico">{icono}</div><div>'
            f'<div class="lab">{etiqueta}</div><div class="num">{valor}</div></div></div>')


c0, c1, c2, c3 = st.columns([1.25, 1, 1, 1])
c0.markdown(hero, unsafe_allow_html=True)
c1.markdown(tarjeta("📦", "Pedidos Hoy", len(pedidos)), unsafe_allow_html=True)
c2.markdown(tarjeta("⏱️", "ETA Promedio", f"{pedidos['ETA'].mean():.2f} h"), unsafe_allow_html=True)
c3.markdown(tarjeta("⚠️", "Riesgo", f"{len(altos)} altos"), unsafe_allow_html=True)
st.caption(f"Riesgo alto: ETA mayor a {umbral} horas")

st.markdown('<div class="seccion">Gráficos</div>', unsafe_allow_html=True)
g1, g2 = st.columns(2)

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
    fig.update_traces(textinfo="value+percent", textfont_color="white")
    fig.update_layout(legend=dict(orientation="h", y=-0.05))
    mostrar_grafico(fig)

t1, t2 = st.columns(2)

with t1:
    st.markdown('<div class="seccion">Tabla de pedidos</div>', unsafe_allow_html=True)
    tabla = pedidos[["Pedido", "ETA", "Riesgo", "Estado"]].style.format({"ETA": "{:.2f}"})
    tabla = (tabla.map if hasattr(tabla, "map") else tabla.applymap)(color_riesgo, subset=["Riesgo"])
    st.dataframe(tabla, hide_index=True, height=330)

with t2:
    st.markdown('<div class="seccion">Tendencia</div>', unsafe_allow_html=True)
    tend = pedidos.groupby("Fecha", as_index=False)["ETA"].mean()
    tend["ETA"] = tend["ETA"].round(2)
    fig = px.line(tend, x="Fecha", y="ETA", markers=True, title="Tendencia semanal del ETA (h)")
    fig.update_traces(line_color=TURQUESA, line_width=3, marker_size=9,
                      fill="tozeroy", fillcolor="rgba(20,184,166,0.18)")
    fig.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickformat="%d %b")
    mostrar_grafico(fig)
