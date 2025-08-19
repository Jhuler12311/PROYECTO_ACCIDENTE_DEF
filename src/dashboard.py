import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# Importar modelos
from modelos.modelo_accidentes import ModeloAccidentesRapido, ModeloAccidentesCompleto

# Configuración inicial
st.set_page_config(page_title="Dashboard Accidentes CR", layout="wide")

# === Buscar dataset generado ===
def encontrar_csv(nombre_archivo, max_depth=3):
    base_path = Path(__file__).resolve().parent
    for depth in range(max_depth + 1):
        for ruta in base_path.rglob(nombre_archivo):
            if ruta.relative_to(base_path).parts[-1] == nombre_archivo:
                return ruta
        base_path = base_path.parent
    return None

DATA_PATH = encontrar_csv("accidentes_clima_2023.csv")

if DATA_PATH is None:
    st.error("❌ No se encontró el archivo procesado. Ejecuta `main.py` para generarlo.")
    st.stop()

# === Cargar datos ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
        if df["hora"].dtype == object:
            df["hora"] = pd.to_datetime(df["hora"], format="%H:%M").dt.hour
        return df
    except Exception as e:
        st.error(f"⚠️ Error al cargar datos: {str(e)}")
        st.stop()

df = load_data()
st.write("🗂️ Columnas disponibles en el dataset:", df.columns.tolist())

# === Sidebar con filtros ===
with st.sidebar:
    st.header("Filtros")
    prov_sel = st.selectbox("Provincia", ["Todas"] + sorted(df["provincia"].unique().tolist()), key="provincia_filtros")
    tipo_accidente = st.multiselect("Tipo de accidente", df["tipo de accidente"].unique(), key="tipo_accidente_filtros")
    rango_fechas = st.date_input("Rango de fechas", [df["fecha"].min(), df["fecha"].max()], key="rango_fechas_filtros")

# Aplicar filtros
df_filtrado = df.copy()
if prov_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["provincia"] == prov_sel]
if tipo_accidente:
    df_filtrado = df_filtrado[df_filtrado["tipo de accidente"].isin(tipo_accidente)]
if len(rango_fechas) == 2:
    df_filtrado = df_filtrado[
        (df_filtrado["fecha"] >= pd.to_datetime(rango_fechas[0])) &
        (df_filtrado["fecha"] <= pd.to_datetime(rango_fechas[1]))
    ]

# === KPIs ===
st.subheader("📊 Métricas Clave")
cols = st.columns(4)
cols[0].metric("Total Accidentes", len(df_filtrado))
cols[1].metric("Días con lluvia", df_filtrado[df_filtrado["precip_acum"] >= 1.0]["fecha"].nunique())
cols[2].metric("Hora pico", f"{df_filtrado['hora'].mode()[0]}:00")
cols[3].metric("Provincia principal", df_filtrado["provincia"].mode()[0])

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs([
    "Análisis Temporal", "Análisis Climático", "Datos", "Predicción de Accidentes"
])

# --- TAB 1: Análisis Temporal ---
with tab1:
    acc_hora = df_filtrado.groupby("hora").size().reset_index(name="accidentes")
    fig_hora = px.bar(acc_hora, x="hora", y="accidentes", title="Accidentes por hora del día")
    st.plotly_chart(fig_hora, use_container_width=True)

    df_filtrado["dia_semana"] = df_filtrado["fecha"].dt.day_name()
    acc_dia = df_filtrado.groupby("dia_semana").size().reset_index(name="accidentes")
    fig_dia = px.bar(
        acc_dia, x="dia_semana", y="accidentes",
        category_orders={"dia_semana": ["Monday", "Tuesday", "Wednesday",
                                        "Thursday", "Friday", "Saturday", "Sunday"]},
        title="Accidentes por día de la semana"
    )
    st.plotly_chart(fig_dia, use_container_width=True)

# --- TAB 2: Análisis Climático ---
with tab2:
    df_filtrado["lluvia"] = df_filtrado["precip_acum"].apply(lambda x: "Con lluvia" if x >= 1.0 else "Sin lluvia")
    acc_lluvia = df_filtrado.groupby("lluvia").size().reset_index(name="accidentes")
    fig_lluvia = px.pie(acc_lluvia, values="accidentes", names="lluvia",
                        title="Proporción de accidentes con/sin lluvia")
    st.plotly_chart(fig_lluvia, use_container_width=True)

    if "clase de accidente" in df_filtrado.columns:
        fig_gravedad = px.box(df_filtrado, x="clase de accidente", y="precip_acum",
                              title="Distribución de precipitación por tipo de accidente")
        st.plotly_chart(fig_gravedad, use_container_width=True)

# --- TAB 3: Datos ---
with tab3:
    st.dataframe(
        df_filtrado.sort_values("fecha", ascending=False),
        column_config={
            "fecha": st.column_config.DateColumn("Fecha"),
            "hora": st.column_config.NumberColumn("Hora", format="%d"),
            "precip_acum": st.column_config.NumberColumn("Lluvia (mm)", format="%.2f")
        },
        hide_index=True,
        use_container_width=True
    )

# --- TAB 4: Predicción ---
with tab4:
    st.subheader("🔮 Predicción de Accidentes")

    # === Entrenar modelos (solo una vez) ===
    @st.cache_resource
    def entrenar_modelos(df_ent):
        m1 = ModeloAccidentesRapido()
        m1.entrenar(df_ent)

        m2 = ModeloAccidentesCompleto()
        m2.entrenar(df_ent)
        return m1, m2

    modelo_rapido, modelo_completo = entrenar_modelos(df)

    # === Inputs de usuario ===
    hora = st.slider("Hora del día", 0, 23, 12, key="hora_pred")
    dia_semana = st.selectbox(
        "Día de la semana",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][x],
        key="dia_semana_pred"
    )
    provincia = st.selectbox("Provincia", sorted(df["provincia"].unique()), key="provincia_pred")
    precip = st.number_input("Precipitación acumulada (mm)", min_value=0.0, value=0.0, step=0.5, key="precip_pred")
    tipo_via = st.selectbox("Tipo de vía", df["tipo_via"].dropna().unique(), key="tipo_via_pred")
    estado_tiempo = st.selectbox("Estado del tiempo", df["estado del tiempo"].dropna().unique(), key="estado_tiempo_pred")
    estado_calzada = st.selectbox("Estado de la calzada", df["estado de la calzada"].dropna().unique(), key="estado_calzada_pred")

    if st.button("Predecir 🚦"):
        # Predicciones simples
        pred1 = modelo_rapido.predecir(hora, dia_semana, provincia, precip,
                                       tipo_via, estado_tiempo, estado_calzada)
        pred2 = modelo_completo.predecir(hora, dia_semana, provincia, precip,
                                         tipo_via, estado_tiempo, estado_calzada)

        st.success(f"✅ **Modelo rápido predice:** {pred1}")
        st.success(f"✅ **Modelo completo predice:** {pred2}")

        # === Probabilidades de ambos modelos ===
        df_probs1 = modelo_rapido.probabilidades(hora, dia_semana, provincia, precip,
                                                 tipo_via, estado_tiempo, estado_calzada)
        df_probs1["Modelo"] = "Rápido"

        df_probs2 = modelo_completo.probabilidades(hora, dia_semana, provincia, precip,
                                                   tipo_via, estado_tiempo, estado_calzada)
        df_probs2["Modelo"] = "Completo"

        df_probs = pd.concat([df_probs1, df_probs2])

        # Gráfico comparativo de probabilidades
        fig_probs = px.bar(
            df_probs, x="Tipo de accidente", y="Probabilidad", color="Modelo",
            barmode="group", text=df_probs["Probabilidad"].apply(lambda p: f"{p:.1%}"),
            title="Probabilidades por tipo de accidente (comparación de modelos)"
        )
        st.plotly_chart(fig_probs, use_container_width=True)

st.success(f"✅ Datos cargados correctamente desde: {DATA_PATH}")
