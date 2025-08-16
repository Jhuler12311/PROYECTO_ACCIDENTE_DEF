import pandas as pd
from pathlib import Path
import sys

class GestorDatos:
    def __init__(self, ruta_raw="data/raw", ruta_processed="data/processed", logger=None):
        self.ruta_raw = Path(ruta_raw)
        self.ruta_processed = Path(ruta_processed)
        self.logger = logger

    def cargar_csv_accidentes(self, nombre_archivo: str) -> pd.DataFrame:
        ruta = self.ruta_raw / nombre_archivo

        if not ruta.exists():
            print(f"❌ No se encontró el archivo {ruta}")
            sys.exit(1)

        # Leer con separador ; y manejo de acentos
        try:
            df = pd.read_csv(ruta, sep=";", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(ruta, sep=";", encoding="latin-1")

        print(f"✅ CSV cargado con {len(df)} registros y {len(df.columns)} columnas")
        print("Columnas detectadas:", df.columns.tolist())

        return df

    def limpiar_accidentes(self, df: pd.DataFrame) -> pd.DataFrame:
        # Renombrar columna "Provincia " quitando espacios
        df.columns = [c.strip() for c in df.columns]

        # Crear columna fecha a partir de Año, Mes y Día
        if all(col in df.columns for col in ["Año", "Mes", "Día"]):
            df["fecha"] = pd.to_datetime(df[["Año", "Mes", "Día"]], errors="coerce")

        # Asegurar hora como entero
        if "Hora" in df.columns:
            df["hora"] = pd.to_numeric(df["Hora"], errors="coerce").fillna(-1).astype(int)

        # Normalizar provincia
        if "Provincia" in df.columns:
            df["provincia"] = df["Provincia"].astype(str).str.title().str.strip()

        # Tipo de vía → no está directo, pero podríamos usar "Tipo ruta"
        if "Tipo ruta" in df.columns:
            df["tipo_via"] = df["Tipo ruta"].astype(str).str.title().str.strip()
        else:
            df["tipo_via"] = "Desconocido"

        # Quitar registros sin fecha o provincia
        df = df.dropna(subset=["fecha", "provincia"])

        print(f"✅ Dataset limpio con {len(df)} registros")
        return df