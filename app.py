import re
import itertools
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import requests
import streamlit as st

OFFICIAL_URL = "https://www.baloto.com/resultados"

st.set_page_config(page_title="Baloto Predictor", page_icon="🎯", layout="wide")

st.title("🎯 Baloto Predictor — Analytics")
st.caption("Análisis estadístico histórico y generación de combinaciones. No existe garantía de predicción de un sorteo aleatorio.")

@st.cache_data(ttl=3600)
def load_official_history(max_pages=126):
    rows = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for page in range(1, max_pages + 1):
        url = OFFICIAL_URL if page == 1 else f"{OFFICIAL_URL}?page={page}"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception:
            continue
        # Busca fechas colombianas y seis números en las filas de resultados.
        months = ("Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre")
        pat = re.compile(
            rf"(\d{{1,2}} de (?:{months}) de \d{{4}}).*?"
            r"(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{1,2})",
            re.S | re.I,
        )
        text = re.sub(r"\s+", " ", r.text)
        for m in pat.finditer(text):
            date = m.group(1)
            nums = list(map(int, m.groups()[1:]))
            if len(set(nums[:5])) == 5 and all(1 <= x <= 43 for x in nums[:5]) and 1 <= nums[5] <= 16:
                rows.append({"fecha": date, "n1": nums[0], "n2": nums[1], "n3": nums[2],
                             "n4": nums[3], "n5": nums[4], "superbalota": nums[5]})
    df = pd.DataFrame(rows).drop_duplicates()
    if not df.empty:
        df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
    return df

def generate_demo():
    # Datos mínimos de respaldo para que la app abra incluso si el sitio oficial bloquea scraping.
    data = [
        ("2026-09-26",[11,15,16,20,36],12),
        ("2026-09-23",[4,9,14,15,16],12),
        ("2026-09-21",[12,13,21,40,42],8),
        ("2026-09-19",[11,12,17,28,31],15),
        ("2026-09-16",[4,8,9,39,41],14),
        ("2026-09-14",[16,25,27,34,43],11),
        ("2026-09-12",[3,5,13,17,28],1),
        ("2026-09-10",[7,29,31,32,38],13),
    ]
    return pd.DataFrame(
        [{"fecha": pd.Timestamp(d), **{f"n{i+1}": v[i] for i in range(5)}, "superbalota": sb}
         for d,v,sb in data]
    )

@st.cache_data(ttl=3600)
def get_data():
    df = load_official_history()
    if len(df) < 20:
        return generate_demo(), True
    return df, False

df, demo = get_data()

if demo:
    st.warning("No se pudo cargar suficiente histórico desde el sitio oficial. Se está mostrando un pequeño conjunto de respaldo. Pulsa Actualizar más tarde.")

st.sidebar.header("Configuración")
window = st.sidebar.slider("Ventana reciente", 10, min(200, max(10, len(df))), min(50, max(10, len(df))))
n_jugadas = st.sidebar.slider("Número de combinaciones", 1, 30, 10)
seed = st.sidebar.number_input("Semilla reproducible", 0, 999999, 2026)

main_cols = [f"n{i}" for i in range(1,6)]
recent = df.tail(window)

def stats_table(data):
    counts = Counter(data[main_cols].to_numpy().ravel())
    last_seen = {}
    for n in range(1,44):
        idx = data.index[(data[main_cols] == n).any(axis=1)]
        last_seen[n] = (len(data)-1 - data.index.get_loc(idx[-1])) if len(idx) else len(data)
    rows=[]
    for n in range(1,44):
        freq=counts[n]
        expected=len(data)*5/43
        z=(freq-expected)/math.sqrt(max(expected*(1-5/43),1e-9))
        rows.append([n,freq,last_seen[n],z])
    return pd.DataFrame(rows, columns=["Número","Frecuencia","Sorteos_desde_última","Z"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard","🔥 Ranking","🎯 Generador","🧪 Backtesting"])

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sorteos cargados", len(df))
    c2.metric("Ventana analizada", len(recent))
    c3.metric("Números principales", "1–43")
    c4.metric("Superbalota", "1–16")

    st.subheader("Frecuencia de números")
    st.bar_chart(stats_table(recent).set_index("Número")["Frecuencia"])

    st.subheader("Superbalota")
    sb = recent["superbalota"].value_counts().reindex(range(1,17), fill_value=0)
    st.bar_chart(sb)

with tab2:
    s = stats_table(recent)
    s["Score"] = (
        0.45 * (s["Frecuencia"] / max(s["Frecuencia"].max(),1)) +
        0.35 * (s["Sorteos_desde_última"] / max(s["Sorteos_desde_última"].max(),1)) +
        0.20 * ((s["Z"] - s["Z"].min()) / max(s["Z"].max()-s["Z"].min(),1e-9))
    )
    st.write("Puntuación heurística: combina frecuencia reciente, atraso y desviación estadística. No es una probabilidad de ganar.")
    st.dataframe(s.sort_values("Score", ascending=False).round(3), use_container_width=True, hide_index=True)

with tab3:
    s = stats_table(recent)
    freq = s.set_index("Número")["Frecuencia"].astype(float)
    # Suavizado para evitar probabilidades cero.
    weights = (freq + 1.0).to_numpy()
    weights = weights / weights.sum()
    rng = np.random.default_rng(int(seed))

    combos=[]
    seen=set()
    while len(combos) < n_jugadas:
        pick = tuple(sorted(rng.choice(np.arange(1,44), size=5, replace=False, p=weights)))
        if pick in seen:
            continue
        seen.add(pick)
        sb_counts = recent["superbalota"].value_counts().reindex(range(1,17), fill_value=0).astype(float)
        sb_weights=(sb_counts+1).to_numpy(); sb_weights/=sb_weights.sum()
        sb=int(rng.choice(np.arange(1,17), p=sb_weights))
        score=float(sum(freq[x] for x in pick))
        combos.append({"Jugada":len(combos)+1,"Números":" - ".join(f"{x:02d}" for x in pick),
                       "Superbalota":f"{sb:02d}","Score_modelo":round(score,2)})
    out=pd.DataFrame(combos)
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Descargar combinaciones CSV", out.to_csv(index=False).encode("utf-8"), "baloto_jugadas.csv", "text/csv")

with tab4:
    st.write("Backtesting simple: cada sorteo se estima usando únicamente los sorteos anteriores y se mide cuántos aciertos obtiene la combinación ponderada.")
    min_train = st.slider("Sorteos iniciales de entrenamiento", 10, max(10, len(df)-1), min(30, max(10,len(df)-1)))
    trials = st.slider("Número de sorteos a evaluar", 1, max(1,len(df)-min_train), min(50,max(1,len(df)-min_train)))
    if len(df) <= min_train:
        st.info("Se necesita más histórico para ejecutar el backtesting.")
    else:
        rows=[]
        rng=np.random.default_rng(int(seed))
        end=min(len(df), min_train+trials)
        for i in range(min_train,end):
            train=df.iloc[:i]
            target=set(df.iloc[i][main_cols].tolist())
            counts=Counter(train[main_cols].to_numpy().ravel())
            w=np.array([counts[n]+1 for n in range(1,44)],dtype=float); w/=w.sum()
            pick=set(rng.choice(np.arange(1,44),5,replace=False,p=w))
            rows.append({"Fecha":df.iloc[i]["fecha"],"Aciertos":len(pick & target),"Combinación":" - ".join(map(str,sorted(pick)))})
        bt=pd.DataFrame(rows)
        st.metric("Promedio de aciertos por jugada", round(bt["Aciertos"].mean(),3))
        st.metric("Máximo de aciertos observado", int(bt["Aciertos"].max()))
        st.dataframe(bt,use_container_width=True,hide_index=True)

st.divider()
st.caption("Fuente del histórico: sitio oficial de Baloto. El análisis es informativo y no constituye una predicción garantizada ni una recomendación de juego.")
