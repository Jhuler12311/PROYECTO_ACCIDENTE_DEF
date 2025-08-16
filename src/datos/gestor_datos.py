import pandas as pd
from pathlib import Path
import sys

class GestorDatos:
    def __init__(self, ruta_raw="data/raw", ruta_processed="data/processed", logger=None):
        self.ruta_raw = Path(ruta_raw)
        self.ruta_processed = Path(ruta_processed)
        self.logger = logger

    def cargar_csv_accidentes(self, nombre_archivo: str) -> pd.DataFrame:
        """
        Carga el CSV de accidentes desde ruta absoluta o relativa.
        Soporta separador ; y codificación utf-8 o latin-1.
        """
        ruta = Path(nombre_archivo) if Path(nombre_archivo).is_absolute() else self.ruta_raw / nombre_archivo

        if not ruta.exists():
            print(f"❌ No se encontró el archivo {ruta}")
            sys.exit(1)

        try:
            df = pd.read_csv(ruta, sep=";", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(ruta, sep=";", encoding="latin-1")

        print(f"✅ CSV cargado: {len(df)} registros y {len(df.columns)} columnas")
        print(f"📌 Columnas detectadas: {df.columns.tolist()}")
        return df

    def limpiar_accidentes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y normaliza el dataset de accidentes:
        - Quita espacios en nombres de columnas y pasa a minúsculas
        - Renombra columnas con caracteres especiales mal codificados
        - Extrae día y mes en formato numérico
        - Crea columna 'fecha' a partir de Año, Mes, Día
        - Convierte Hora a entero
        - Normaliza provincia y tipo de vía
        """

        # Normalizar nombres de columnas
        df.columns = df.columns.str.strip().str.lower()

        # Renombrar columnas con problemas de codificación
        rename_map = {
            "aã±o": "año",
            "dã\xada": "día",
            "mes ": "mes",
            "provincia ": "provincia"
        }
        df.rename(columns=rename_map, inplace=True)

        # Limpiar columna día (extraer número)
        if "día" in df.columns:
            df["día"] = df["día"].astype(str).str.extract(r"(\d+)").astype(float)

        # Mapear meses a número
        meses_map = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
        }
        if "mes" in df.columns:
            df["mes"] = (
                df["mes"].astype(str)
                .str.extract(r"[A-ZÁÉÍÓÚ]\.\s*([A-Za-zÁÉÍÓÚáéíóú]+)")[0]
                .str.lower()
                .map(meses_map)
            )

        # Crear columna fecha
        if all(col in df.columns for col in ["año", "mes", "día"]):
            df["fecha"] = pd.to_datetime(
                dict(
                    year=pd.to_numeric(df["año"], errors="coerce"),
                    month=pd.to_numeric(df["mes"], errors="coerce"),
                    day=pd.to_numeric(df["día"], errors="coerce")
                ),
                errors="coerce"
            )
        else:
            print("⚠️ No se encontraron columnas de fecha completas.")

        # Crear columna hora (primer número del rango)
        if "hora" in df.columns:
            df["hora"] = df["hora"].astype(str).str.extract(r"(\d{1,2})").astype(float).fillna(-1).astype(int)

        # Normalizar provincia
        if "provincia" in df.columns:
            df["provincia"] = df["provincia"].astype(str).str.title().str.strip()
        else:
            df["provincia"] = "Desconocido"

        # Determinar tipo de vía usando 'tipo ruta'
        if "tipo ruta" in df.columns:
            df["tipo_via"] = df["tipo ruta"].astype(str).str.title().str.strip()
        else:
            df["tipo_via"] = "Desconocido"

        # Mantener solo columnas relevantes si existen
        columnas_relevantes = [
            "fecha", "hora", "provincia", "tipo_via",
            "clase de accidente", "tipo de accidente",
            "estado del tiempo", "ruta", "kilómetro"
        ]
        columnas_presentes = [c for c in columnas_relevantes if c in df.columns]
        df = df[columnas_presentes]

        # Eliminar registros sin fecha o provincia
        df = df[df["fecha"].notna() & df["provincia"].notna()]

        print(f"✅ Dataset limpio con {len(df)} registros y {len(df.columns)} columnas")
        return df

    def unir_con_clima(self, df_accidentes: pd.DataFrame, df_clima: pd.DataFrame) -> pd.DataFrame:
        """
        Une el dataset de accidentes con el dataset de clima por fecha y provincia.
        """
        if not all(col in df_clima.columns for col in ["fecha", "provincia", "precip_acum"]):
            print("❌ El DataFrame de clima no tiene las columnas necesarias.")
            return df_accidentes

        df_final = df_accidentes.merge(df_clima, on=["fecha", "provincia"], how="left")
        print(f"✅ Dataset combinado: {len(df_final)} registros")
        return df_final

    def guardar_processed(self, df: pd.DataFrame, nombre_salida: str):
        """
        Guarda un DataFrame en la carpeta processed.
        """
        ruta_salida = self.ruta_processed / nombre_salida
        self.ruta_processed.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta_salida, index=False)
        print(f"💾 Dataset guardado en: {ruta_salida}")
