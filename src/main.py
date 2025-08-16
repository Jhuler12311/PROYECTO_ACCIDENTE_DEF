from datos.gestor_datos import GestorDatos
from api.cliente_api import ClienteAPI
from eda.procesador_eda import ProcesadorEDA
from datetime import date

def main():
    # === 1) Inicializar gestor de datos ===
    gd = GestorDatos()

    # === 2) Cargar CSV de accidentes (ruta absoluta para evitar errores) ===
    df_acc = gd.cargar_csv_accidentes(
        r"C:\Users\98248\Downloads\PYCHAR\proyecto_accidentes\proyecto_accidentes\data\raw\accidentes_2023.csv"
    )

    # === 3) Limpiar dataset de accidentes ===
    df_acc_limpio = gd.limpiar_accidentes(df_acc)

    # === 4) Descargar datos de clima para todas las provincias ===
    api = ClienteAPI()
    df_clima = api.clima_todas_provincias(
        start=date(2023, 1, 1),
        end=date(2023, 12, 31)
    )

    # === 5) Unir accidentes con clima ===
    df_final = gd.unir_con_clima(df_acc_limpio, df_clima)

    # === 6) Guardar dataset procesado ===
    gd.guardar_processed(df_final, "accidentes_clima_2023.csv")

    # === 7) Análisis exploratorio de datos ===
    eda = ProcesadorEDA()

    print("\n=== Resumen general ===")
    print(eda.resumen_general(df_final))

    print("\n=== Accidentes por provincia ===")
    print(eda.accidentes_por_provincia(df_final))

    print("\n=== Accidentes por hora ===")
    print(eda.accidentes_por_hora(df_final))

    print("\n=== Comparación lluvia vs sin lluvia ===")
    print(eda.comparar_lluvia(df_final, umbral_mm=1.0))


if __name__ == "__main__":
    main()
