from datos.gestor_datos import GestorDatos
from api.cliente_api import ClienteAPI
from eda.procesador_eda import ProcesadorEDA
from modelos.modelo_accidentes import ModeloAccidentesRapido, ModeloAccidentesCompleto
from datetime import date
from pathlib import Path
import sys


def main():
    print("\n🚦 Iniciando pipeline de análisis de accidentes en Costa Rica...\n")

    # === 1) Inicializar gestor de datos ===
    gd = GestorDatos()

    # === 2) Cargar CSV de accidentes ===
    csv_path = Path(
        r"C:\Users\98248\Downloads\PYCHAR\proyecto_accidentes\data\raw\accidentes_2023.csv"
    )

    if not csv_path.exists():
        print(f"❌ No se encontró el archivo de accidentes en {csv_path}")
        sys.exit(1)

    print("📂 Cargando dataset de accidentes...")
    df_acc = gd.cargar_csv_accidentes(csv_path)

    # === 3) Limpiar dataset de accidentes ===
    print("🧹 Limpiando dataset de accidentes...")
    df_acc_limpio = gd.limpiar_accidentes(df_acc)

    # === 4) Descargar datos de clima ===
    print("🌦️ Descargando datos de clima de la API...")
    api = ClienteAPI()
    df_clima = api.clima_todas_provincias(
        start=date(2023, 1, 1),
        end=date(2023, 12, 31)
    )

    # === 5) Unir accidentes con clima ===
    print("🔗 Uniendo accidentes con datos de clima...")
    df_final = gd.unir_con_clima(df_acc_limpio, df_clima)

    # === 6) Guardar dataset procesado ===
    print("💾 Guardando dataset procesado...")
    gd.guardar_processed(df_final, "accidentes_clima_2023.csv")

    # === 7) Análisis exploratorio de datos ===
    print("\n📊 Iniciando análisis exploratorio...")
    eda = ProcesadorEDA()
    print("\n=== Resumen general ===")
    print(eda.resumen_general(df_final))
    print("\n=== Accidentes por provincia ===")
    print(eda.accidentes_por_provincia(df_final))
    print("\n=== Accidentes por hora ===")
    print(eda.accidentes_por_hora(df_final))
    print("\n=== Comparación lluvia vs sin lluvia ===")
    print(eda.comparar_lluvia(df_final, umbral_mm=1.0))

    # === 8) Entrenar modelos ===
    print("\n=== 🔹 Entrenando modelo RÁPIDO (HistGradientBoosting) ===")
    modelo_rapido = ModeloAccidentesRapido()
    modelo_rapido.entrenar(df_final, validar=True)

    print("\n=== 🔹 Entrenando modelo COMPLETO (GradientBoosting) ===")
    modelo_completo = ModeloAccidentesCompleto()
    modelo_completo.entrenar(df_final, validar=True)

    # === 9) Guardar resultados en reports/ ===
    print("\n📝 Guardando reportes de resultados...")
    reports_path = Path("reports")
    reports_path.mkdir(parents=True, exist_ok=True)

    # Ejemplo de predicción
    ejemplo = dict(
        hora=18,
        dia_semana=4,   # Viernes
        provincia="San José",
        precip_acum=7.0,
        tipo_via="Ruta Nacional",
        estado_tiempo="Lluvia",
        estado_calzada="Mojada"
    )

    pred_rapido = modelo_rapido.predecir(**ejemplo)
    pred_completo = modelo_completo.predecir(**ejemplo)

    with open(reports_path / "precision_modelos.txt", "w", encoding="utf-8") as f:
        f.write("📊 Resultados de entrenamiento:\n\n")
        f.write("- Modelo RÁPIDO (HistGradientBoosting): entrenamiento veloz, precisión media reportada en consola.\n")
        f.write("- Modelo COMPLETO (GradientBoosting): entrenamiento más lento, mejor precisión esperada.\n\n")
        f.write("🔮 Predicciones de ejemplo (entrada fija):\n")
        f.write(f"   • Modelo rápido: {pred_rapido}\n")
        f.write(f"   • Modelo completo: {pred_completo}\n")
        f.write("\n👉 Ver consola para métricas detalladas.\n")

    # === 10) Mostrar predicciones en consola ===
    print("\n=== Ejemplo de predicción ===")
    print(f"🔮 Modelo rápido predice: {pred_rapido}")
    print(f"🔮 Modelo completo predice: {pred_completo}")

    print("\n✅ Pipeline finalizado correctamente.")


if __name__ == "__main__":
    main()
