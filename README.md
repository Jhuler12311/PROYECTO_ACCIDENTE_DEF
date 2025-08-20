# 🚦 Proyecto de Predicción de Accidentes de Tránsito en Costa Rica

Este proyecto utiliza **Big Data + Machine Learning** para analizar y predecir accidentes de tránsito en Costa Rica entre 2020 y 2024.  
El sistema integra datos de **accidentes históricos** con datos de **clima en tiempo real** y presenta los resultados en un **dashboard interactivo (Streamlit)**.

---

## 📂 Estructura del proyecto

```
proyecto_accidentes/
│── data/               # Archivos de datos
│   ├── raw/            # Datos originales (CSV)
│   ├── processed/      # Datos procesados listos para ML
│── reports/            # Resultados, métricas y logs
│── src/                # Código fuente
│   ├── api/            # Cliente de API para clima
│   ├── datos/          # Gestión de datos (ETL)
│   ├── eda/            # Análisis exploratorio
│   ├── modelos/        # Modelos ML (scikit-learn)
│   ├── main.py         # Pipeline principal
│   ├── dashboard.py    # Dashboard Streamlit
│── requirements.txt    # Dependencias del proyecto
│── README.md           # Documentación
│── .gitignore
```

---

## ⚙️ Instalación y requisitos

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Jhuler12311/PROYECTO_ACCIDENTE_DEF.git
   cd PROYECTO_ACCIDENTE_DEF
   ```

2. Crear entorno virtual e instalar dependencias:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # En Windows
   source .venv/bin/activate  # En Linux/Mac

   pip install -r requirements.txt
   ```

---

## ▶️ Ejecución del proyecto

### 1. Preprocesar datos y entrenar modelos
```bash
python src/main.py
```

Este script:
- Limpia los datos de accidentes.
- Descarga y procesa datos climáticos.
- Une datasets en `data/processed/`.
- Entrena modelos de predicción.
- Genera reportes en `reports/`.

📌 Resultado esperado:  
Un archivo `data/processed/accidentes_clima_2023.csv` y métricas de precisión en `reports/precision_modelos.txt`.

---

### 2. Ejecutar el Dashboard (Streamlit)
```bash
streamlit run src/dashboard.py
```

📊 El dashboard incluye:
- Accidentes por **hora del día** y **día de la semana**.
- Comparación de accidentes con/sin lluvia 🌧️.
- Accidentes por provincia 📍.
- Predicción del **tipo de accidente** a partir de condiciones (hora, clima, provincia, etc.).

---

## 🖼️ Capturas de pantalla

### 🔹 Dashboard principal
![Dashboard Accidentes](https://via.placeholder.com/900x400.png?text=Dashboard+Accidentes+CR)

### 🔹 Predicción de accidentes
![Predicción](https://via.placeholder.com/900x400.png?text=Prediccion+Accidentes)

*(Sustituye estas imágenes por tus propias capturas cuando ejecutes el dashboard.)*

---

## 🔮 Ejemplo de predicción (código)

```python
from modelos.modelo_accidentes import ModeloAccidentesRapido

# Cargar datos procesados
import pandas as pd
df = pd.read_csv("data/processed/accidentes_clima_2023.csv", parse_dates=["fecha"])

# Entrenar modelo rápido
modelo = ModeloAccidentesRapido()
modelo.entrenar(df)

# Ejemplo de predicción
pred = modelo.predecir(
    hora=18,
    dia_semana=4,   # Viernes
    provincia="San José",
    precip_acum=7.0,
    tipo_via="Ruta Nacional",
    estado_tiempo="Lluvia",
    estado_calzada="Mojada"
)

print(f"🔮 Predicción: {pred}")
```

📌 **Salida esperada**:  
```
🔮 Predicción: Choque múltiple
```

---

## 📊 Tecnologías utilizadas
- **Python 3.10+**
- **scikit-learn** → Modelos ML (GradientBoosting, HistGradientBoosting)
- **pandas / numpy** → Procesamiento de datos
- **requests** → Cliente API clima
- **plotly + streamlit** → Dashboard interactivo
- **GitHub** → Control de versiones

---

## ✨ Autor
- **Jhuler12311** 👨‍💻  
*(Proyecto académico – Predicción de Accidentes CR)*
