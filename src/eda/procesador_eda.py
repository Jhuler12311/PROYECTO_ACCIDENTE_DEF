import pandas as pd


class ProcesadorEDA:
    def resumen_general(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve un resumen estadístico del dataset.
        """
        print("📊 Resumen general del dataset:")
        return df.describe(include="all")

    def accidentes_por_provincia(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cuenta la cantidad de accidentes por provincia.
        """
        if "provincia" not in df.columns:
            print("❌ No se encontró la columna 'provincia'")
            return pd.DataFrame()

        tabla = df.groupby("provincia", as_index=False).size().rename(columns={"size": "accidentes"})
        print("📍 Accidentes por provincia calculados.")
        return tabla.sort_values(by="accidentes", ascending=False)

    def accidentes_por_hora(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cuenta la cantidad de accidentes por hora del día.
        """
        if "hora" not in df.columns:
            print("❌ No se encontró la columna 'hora'")
            return pd.DataFrame()

        tabla = df.groupby("hora", as_index=False).size().rename(columns={"size": "accidentes"})
        print("🕒 Accidentes por hora calculados.")
        return tabla.sort_values(by="hora")

    def comparar_lluvia(self, df: pd.DataFrame, umbral_mm=1.0) -> pd.DataFrame:
        """
        Compara cantidad de accidentes en días con lluvia vs sin lluvia.
        Un día se considera 'con lluvia' si precip_acum >= umbral_mm.
        """
        if "precip_acum" not in df.columns:
            print("❌ No se encontró la columna 'precip_acum'")
            return pd.DataFrame()

        df_temp = df.copy()
        df_temp["lluvia"] = df_temp["precip_acum"].fillna(0).apply(
            lambda x: "Con lluvia" if x >= umbral_mm else "Sin lluvia")

        tabla = df_temp.groupby("lluvia", as_index=False).size().rename(columns={"size": "accidentes"})
        print("🌧 Comparación lluvia vs sin lluvia calculada.")
        return tabla
