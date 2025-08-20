import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# === CONFIGURACIÓN INICIAL ===
st.set_page_config(page_title="Dashboard Accidentes CR", layout="wide")

# === ESTADO DE LA SESIÓN ===
if 'map_initialized' not in st.session_state:
    st.session_state.map_initialized = False
if 'last_map_bounds' not in st.session_state:
    st.session_state.last_map_bounds = None
if 'map_data' not in st.session_state:
    st.session_state.map_data = None


# === MODELOS DE PREDICCIÓN ===
class ModeloAccidentesRapido:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.label_encoders = {}

    def entrenar(self, df):
        # Preparar características
        X = df[['hora', 'precip_acum']].copy()
        y = df['tipo de accidente']

        # Entrenar modelo
        self.model.fit(X, y)

    def predecir(self, hora, dia_semana, provincia, precip, tipo_via, estado_tiempo, estado_calzada):
        # Para el modelo rápido, solo usamos hora y precipitación
        X_pred = pd.DataFrame([[hora, precip]],
                              columns=['hora', 'precip_acum'])
        return self.model.predict(X_pred)[0]

    def probabilidades(self, hora, dia_semana, provincia, precip, tipo_via, estado_tiempo, estado_calzada):
        X_pred = pd.DataFrame([[hora, precip]],
                              columns=['hora', 'precip_acum'])
        probs = self.model.predict_proba(X_pred)[0]
        return pd.DataFrame({
            'Tipo de accidente': self.model.classes_,
            'Probabilidad': probs
        })


class ModeloAccidentesCompleto:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}

    def entrenar(self, df):
        # Preparar características
        X = df[['hora', 'precip_acum', 'provincia', 'tipo_via', 'estado del tiempo', 'estado de la calzada']].copy()
        y = df['tipo de accidente']

        # Codificar variables categóricas
        for col in ['provincia', 'tipo_via', 'estado del tiempo', 'estado de la calzada']:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))

        # Entrenar modelo
        self.model.fit(X, y)

    def predecir(self, hora, dia_semana, provincia, precip, tipo_via, estado_tiempo, estado_calzada):
        # Preparar datos para predicción
        X_pred = pd.DataFrame([[hora, precip, provincia, tipo_via, estado_tiempo, estado_calzada]],
                              columns=['hora', 'precip_acum', 'provincia', 'tipo_via', 'estado del tiempo',
                                       'estado de la calzada'])

        # Codificar variables categóricas
        for col in ['provincia', 'tipo_via', 'estado del tiempo', 'estado de la calzada']:
            X_pred[col] = self.label_encoders[col].transform([str(X_pred[col].iloc[0])])[0]

        return self.model.predict(X_pred)[0]

    def probabilidades(self, hora, dia_semana, provincia, precip, tipo_via, estado_tiempo, estado_calzada):
        # Preparar datos para predicción
        X_pred = pd.DataFrame([[hora, precip, provincia, tipo_via, estado_tiempo, estado_calzada]],
                              columns=['hora', 'precip_acum', 'provincia', 'tipo_via', 'estado del tiempo',
                                       'estado de la calzada'])

        # Codificar variables categóricas
        for col in ['provincia', 'tipo_via', 'estado del tiempo', 'estado de la calzada']:
            X_pred[col] = self.label_encoders[col].transform([str(X_pred[col].iloc[0])])[0]

        probs = self.model.predict_proba(X_pred)[0]
        return pd.DataFrame({
            'Tipo de accidente': self.model.classes_,
            'Probabilidad': probs
        })


# === GENERAR DATOS DE EJEMPLO ===
@st.cache_data
def generar_datos_ejemplo():
    np.random.seed(42)
    n = 1000

    # Fechas para 2023
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = [start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)) for _ in range(n)]

    # Provincias de Costa Rica
    provincias = ["San José", "Alajuela", "Cartago", "Heredia", "Guanacaste", "Puntarenas", "Limón"]

    # Coordenadas aproximadas por provincia (lat, lon)
    coords_provincias = {
        "San José": [9.9281, -84.0907],
        "Alajuela": [10.0163, -84.2116],
        "Cartago": [9.8644, -83.9194],
        "Heredia": [9.9976, -84.1198],
        "Guanacaste": [10.6350, -85.4377],
        "Puntarenas": [9.9763, -84.8384],
        "Limón": [9.9910, -83.0360]
    }

    # Generar datos
    data = {
        "fecha": dates,
        "hora": np.random.randint(0, 24, n),
        "provincia": np.random.choice(provincias, n),
        "tipo de accidente": np.random.choice(["Colisión frontal", "Alcance", "Atropello", "Salida de vía", "Vuelco"],
                                              n),
        "precip_acum": np.round(np.random.exponential(0.5, n), 2),
        "tipo_via": np.random.choice(["Carretera", "Calle", "Avenida", "Autopista"], n),
        "estado del tiempo": np.random.choice(["Despejado", "Lluvioso", "Nublado", "Niebla"], n),
        "estado de la calzada": np.random.choice(["Seca", "Mojada", "Húmeda", "Con barro"], n),
        "clase de accidente": np.random.choice(["Leve", "Moderado", "Grave", "Fatal"], n),
    }

    df = pd.DataFrame(data)

    # Añadir coordenadas basadas en la provincia
    df["lat"] = df["provincia"].map(lambda x: coords_provincias[x][0] + np.random.normal(0, 0.05))
    df["lon"] = df["provincia"].map(lambda x: coords_provincias[x][1] + np.random.normal(0, 0.05))

    return df


# === BUSCAR DATASET ===
def encontrar_csv(nombre_archivo, max_depth=3):
    base_path = Path(__file__).resolve().parent
    for depth in range(max_depth + 1):
        for ruta in base_path.rglob(nombre_archivo):
            if ruta.relative_to(base_path).parts[-1] == nombre_archivo:
                return ruta
        base_path = base_path.parent
    return None


DATA_PATH = encontrar_csv("accidentes_clima_2023.csv")


# === CARGAR DATOS ===
@st.cache_data
def load_data():
    if DATA_PATH is not None:
        try:
            df = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
            if df["hora"].dtype == object:
                df["hora"] = pd.to_datetime(df["hora"], format="%H:%M").dt.hour
            return df
        except Exception as e:
            st.error(f"⚠️ Error al cargar datos: {str(e)}")
            st.stop()
    else:
        st.warning("⚠️ No se encontró el archivo 'accidentes_clima_2023.csv'. Se cargarán datos de ejemplo.")
        return generar_datos_ejemplo()


df = load_data()


# === PREPARAR DATOS PARA EL MAPA (CACHEADO) ===
@st.cache_data
def preparar_datos_mapa(_df, fecha_min, fecha_max, provincia_filtro, tipo_accidente_filtro):
    df_filtrado = _df.copy()

    # Aplicar filtros
    if provincia_filtro != "Todas":
        df_filtrado = df_filtrado[df_filtrado["provincia"] == provincia_filtro]
    if tipo_accidente_filtro:
        df_filtrado = df_filtrado[df_filtrado["tipo de accidente"].isin(tipo_accidente_filtro)]

    # Filtrar por fecha
    df_filtrado = df_filtrado[
        (df_filtrado["fecha"] >= pd.to_datetime(fecha_min)) &
        (df_filtrado["fecha"] <= pd.to_datetime(fecha_max))
        ]

    # Coordenadas promedio por provincia
    coords_provincias = {
        "San José": [9.9281, -84.0907],
        "Alajuela": [10.0163, -84.2116],
        "Cartago": [9.8644, -83.9194],
        "Heredia": [9.9976, -84.1198],
        "Guanacaste": [10.6350, -85.4377],
        "Puntarenas": [9.9763, -84.8384],
        "Limón": [9.9910, -83.0360]
    }

    # Asegurarse de que tenemos coordenadas
    if "lat" not in df_filtrado.columns or "lon" not in df_filtrado.columns:
        df_filtrado["lat"] = df_filtrado["provincia"].map(
            lambda x: coords_provincias.get(x, [9.9281, -84.0907])[0] + np.random.normal(0, 0.05))
        df_filtrado["lon"] = df_filtrado["provincia"].map(
            lambda x: coords_provincias.get(x, [9.9281, -84.0907])[1] + np.random.normal(0, 0.05))

    # Preparar datos para heatmap
    heat_data = [[row['lat'], row['lon']] for index, row in df_filtrado.iterrows()
                 if not pd.isna(row['lat']) and not pd.isna(row['lon'])]

    return heat_data, df_filtrado


# === SIDEBAR CON FILTROS ===
with st.sidebar:
    st.header("Filtros")
    prov_sel = st.selectbox("Provincia", ["Todas"] + sorted(df["provincia"].unique().tolist()), key="provincia_filtros")
    tipo_accidente = st.multiselect("Tipo de accidente", df["tipo de accidente"].unique(), key="tipo_accidente_filtros")
    rango_fechas = st.date_input("Rango de fechas", [df["fecha"].min(), df["fecha"].max()], key="rango_fechas_filtros")

# Aplicar filtros al dataframe principal
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

# === TABS ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Análisis Temporal", "Análisis Climático", "Datos", "Predicción de Accidentes", "Mapa de Calor"
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

# --- TAB 4: Predicción de Accidentes ---
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
    estado_tiempo = st.selectbox("Estado del tiempo", df["estado del tiempo"].dropna().unique(),
                                 key="estado_tiempo_pred")
    estado_calzada = st.selectbox("Estado de la calzada", df["estado de la calzada"].dropna().unique(),
                                  key="estado_calzada_pred")

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

# --- TAB 5: Mapa de Calor ---
with tab5:
    st.subheader("🌍 Mapa de Calor de Accidentes en Costa Rica")

    # Preparar datos para el mapa (usando función cacheada)
    heat_data, df_mapa = preparar_datos_mapa(
        df,
        rango_fechas[0],
        rango_fechas[1],
        prov_sel,
        tipo_accidente
    )

    # Crear mapa base centrado en Costa Rica
    m = folium.Map(
        location=[9.93, -84.08],
        zoom_start=8,
        tiles='OpenStreetMap',
        width='100%',
        height='100%'
    )

    # Añadir heatmap al mapa
    if heat_data:
        HeatMap(
            heat_data,
            radius=12,
            blur=15,
            max_zoom=10,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
        ).add_to(m)

    # Añadir marcadores para las capitales provinciales
    coords_provincias = {
        "San José": [9.9281, -84.0907],
        "Alajuela": [10.0163, -84.2116],
        "Cartago": [9.8644, -83.9194],
        "Heredia": [9.9976, -84.1198],
        "Guanacaste": [10.6350, -85.4377],
        "Puntarenas": [9.9763, -84.8384],
        "Limón": [9.9910, -83.0360]
    }

    for provincia, coord in coords_provincias.items():
        folium.Marker(
            coord,
            popup=provincia,
            tooltip=provincia,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Mostrar el mapa con un contenedor de tamaño fijo
    map_container = st.container()
    with map_container:
        st_folium(
            m,
            width=1200,
            height=600,
            key="mapa_calor"  # Clave única para el componente
        )

# Información adicional
st.sidebar.info("""
**Notas:**
- Los datos mostrados son de ejemplo
- El mapa de calor muestra la concentración de accidentes
- Utiliza los filtros para refinar los resultados
""")

if DATA_PATH:
    st.success(f"✅ Datos cargados correctamente desde: {DATA_PATH}")
else:
    st.info("ℹ️ Se están utilizando datos de ejemplo para la demostración")