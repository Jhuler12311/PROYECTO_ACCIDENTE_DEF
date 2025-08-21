from basedatos.gestor_basedatos import GestorBaseDatos
from datos.gestor_datos import GestorDatos
from api.cliente_api import ClienteAPI
import pandas as pd
from pathlib import Path


def inicializar_base_datos(ruta_csv=None):
    """Función para inicializar la base de datos con datos reales"""

    print("=== INICIALIZACIÓN DE BASE DE DATOS ===")

    # 1. Inicializar gestores
    gestor_db = GestorBaseDatos()
    gestor_datos = GestorDatos()

    # 2. Cargar datos CSV
    if ruta_csv and Path(ruta_csv).exists():
        print(f"Cargando datos desde: {ruta_csv}")
        df = gestor_datos.cargar_csv_accidentes(ruta_csv)
        df = gestor_datos.limpiar_accidentes(df)
    else:
        print("⚠️ No se encontró archivo CSV, usando datos de ejemplo")
        df = generar_datos_ejemplo()

    # 3. Obtener datos climáticos
    print("Obteniendo datos climáticos...")
    cliente_api = ClienteAPI()

    # Obtener rango de fechas
    fecha_min = df['fecha'].min().strftime('%Y-%m-%d')
    fecha_max = df['fecha'].max().strftime('%Y-%m-%d')

    # Obtener datos climáticos por provincia
    provincias = df['provincia'].unique()
    datos_clima = []

    for provincia in provincias:
        # Coordenadas aproximadas por provincia (simplificado)
        coords = {
            "San José": (9.93, -84.08),
            "Alajuela": (10.02, -84.22),
            "Cartago": (9.86, -83.92),
            "Heredia": (10.00, -84.12),
            "Guanacaste": (10.63, -85.44),
            "Puntarenas": (9.98, -84.83),
            "Limón": (9.99, -83.04)
        }

        lat, lon = coords.get(provincia, (9.93, -84.08))
        df_clima_prov = cliente_api.obtener_datos_climaticos(lat, lon, fecha_min, fecha_max)

        if df_clima_prov is not None:
            df_clima_prov['provincia'] = provincia
            datos_clima.append(df_clima_prov)

    if datos_clima:
        df_clima = pd.concat(datos_clima, ignore_index=True)
        # Unir con datos principales
        df = pd.merge(df, df_clima, on=['fecha', 'provincia'], how='left')
        print("✅ Datos climáticos integrados")
    else:
        print("⚠️ No se pudieron obtener datos climáticos")
        df['precip_acum'] = df.get('precip_acum', 0)  # Mantener valores existentes o 0

    # 4. Insertar en base de datos
    print("Insertando datos en la base de datos...")
    exito = gestor_db.insertar_dataframe(df)

    if exito:
        # Mostrar estadísticas
        stats = gestor_db.obtener_estadisticas()
        print(f"\n📊 ESTADÍSTICAS DE LA BASE DE DATOS:")
        print(f"   - Accidentes: {stats.get('accidentes', 0)}")
        print(f"   - Provincias: {stats.get('provincias', 0)}")
        print(f"   - Tipos de accidente: {stats.get('tipos_accidente', 0)}")
        print(f"   - Rango de fechas: {stats.get('rango_fechas', ('N/A', 'N/A'))}")

        print("\n✅ Base de datos inicializada exitosamente!")
        return True
    else:
        print("❌ Error al inicializar la base de datos")
        return False


def generar_datos_ejemplo():
    """Genera datos de ejemplo si no hay CSV disponible"""
    import numpy as np
    from datetime import datetime, timedelta

    np.random.seed(42)
    n = 500

    # Fechas para 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n)]

    # Provincias de Costa Rica
    provincias = ["San José", "Alajuela", "Cartago", "Heredia", "Guanacaste", "Puntarenas", "Limón"]

    datos_ejemplo = {
        "fecha": dates,
        "hora": np.random.randint(0, 24, n),
        "provincia": np.random.choice(provincias, n),
        "tipo de accidente": np.random.choice(["Colisión frontal", "Alcance", "Atropello", "Salida de vía", "Vuelco"],
                                              n),
        "tipo_via": np.random.choice(["Carretera", "Calle", "Avenida", "Autopista"], n),
        "estado del tiempo": np.random.choice(["Despejado", "Lluvioso", "Nublado", "Niebla"], n),
        "estado de la calzada": np.random.choice(["Seca", "Mojada", "Húmeda", "Con barro"], n),
        "clase de accidente": np.random.choice(["Leve", "Moderado", "Grave", "Fatal"], n),
        "precip_acum": np.round(np.random.exponential(0.5, n), 2),
        "ruta": np.random.choice(["Ruta 1", "Ruta 2", "Ruta 27", "Ruta 32"], n),
        "kilómetro": np.round(np.random.uniform(0, 100, n), 1)
    }

    return pd.DataFrame(datos_ejemplo)


if __name__ == "__main__":
    # Puedes especificar la ruta de tu CSV aquí
    ruta_csv = "data/raw/accidentes.csv"  # Cambia por tu ruta real
    inicializar_base_datos(ruta_csv)