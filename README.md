# Dashboard Logístico ML

Predice el tiempo de entrega (ETA) de pedidos con una red neuronal 64-32-1.

Flujo: base SQLite (pedidos_logisticos.db) -> modelo -> API FastAPI (/predict) -> dashboard Streamlit.

Archivos: app.py, api.py, modelo.npz (pesos del modelo), pedidos_logisticos.db, requirements.txt.

Ejecutar en local: pip install -r requirements.txt  y luego  streamlit run app.py
