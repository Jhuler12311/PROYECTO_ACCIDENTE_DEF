import requests
import pandas as pd
from datetime import date

class ClienteAPI:
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self):
        # Coordenadas aproximadas por provincia (capital o punto central)
        self.coord_provincias = {
            "San José": (9.93, -84.08),
            "Alajuela": (10.02, -84.21),
            "Cartago": (9.87, -83.92),
            "Heredia": (10.00, -84.12),
            "Guanacaste": (10.63, -85.44),
            "Puntarenas": (9.97, -84.83),
            "Limón": (9.99, -83.03)
        }

    def clima_precipitacion_diaria(self, lat, lon, start: date, end: date, tz="America/Costa_Rica") -> pd.DataFrame:
        """
        Descarga precipitación diaria para una ubicación dada (lat/lon) y rango de fechas.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "precipitation_sum",
            "timezone": tz
        }
        r = requests.get(self.BASE_URL, params=params, timeout=60)
        r.raise_for_status()

        j = r.json()
        df = pd.DataFrame({
            "fecha": j["daily"]["time"],
            "precip_acum": j["daily"]["precipitation_sum"]
        })
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    def clima_todas_provincias(self, start: date, end: date) -> pd.DataFrame:
        lista_df = []
        for provincia, (lat, lon) in self.coord_provincias.items():
            print(f"🌦 Descargando clima para {provincia}...")
            df_prov = self.clima_precipitacion_diaria(lat, lon, start, end)
            df_prov["provincia"] = provincia
            lista_df.append(df_prov)

        df_clima_total = pd.concat(lista_df, ignore_index=True)
        print(f"✅ Datos de clima combinados: {len(df_clima_total)} registros")
        return df_clima_total  # ← Esto es importante

