import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier


class BaseModeloAccidentes:
    def __init__(self):
        self.enc_provincia = LabelEncoder()
        self.enc_target = LabelEncoder()
        self.model = None

    def _ingenieria_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega variables derivadas útiles para el modelo.
        """
        if "fecha" in df.columns:
            df["dia_semana"] = df["fecha"].dt.weekday
            df["mes"] = df["fecha"].dt.month
            df["fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)

        if "hora" in df.columns:
            df["hora_categoria"] = pd.cut(
                df["hora"],
                bins=[0, 6, 12, 18, 24],
                labels=["Madrugada", "Mañana", "Tarde", "Noche"],
                right=False
            )

        if "precip_acum" in df.columns:
            df["lluvia_categoria"] = pd.cut(
                df["precip_acum"],
                bins=[-0.1, 0.9, 5, 20, 1000],
                labels=["Sin lluvia", "Lluvia leve", "Lluvia moderada", "Lluvia fuerte"]
            )

        return df

    def entrenar(self, df: pd.DataFrame, validar=False):
        """
        Entrena el modelo para predecir el tipo de accidente.
        """
        # Normalizar nombres de columnas
        df = df.rename(columns=lambda x: x.strip().lower())
        df = self._ingenieria_variables(df)

        # Variable objetivo
        if "tipo de accidente" not in df.columns:
            raise ValueError("⚠️ El dataset no contiene la columna 'tipo de accidente'.")
        y = self.enc_target.fit_transform(df["tipo de accidente"].fillna("Desconocido"))

        # Features
        features = [
            "hora", "dia_semana", "mes", "fin_de_semana", "precip_acum",
            "provincia", "tipo_via", "estado del tiempo", "estado de la calzada",
            "hora_categoria", "lluvia_categoria"
        ]
        X = df[[c for c in features if c in df.columns]].copy()

        # Codificación de categorías
        if "provincia" in X.columns:
            X["provincia"] = self.enc_provincia.fit_transform(X["provincia"].astype(str))

        for col in ["tipo_via", "estado del tiempo", "estado de la calzada",
                    "hora_categoria", "lluvia_categoria"]:
            if col in X.columns:
                X[col] = X[col].astype("category").cat.codes

        # Manejo de valores faltantes
        X = X.fillna(-1)

        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Entrenar modelo
        self.model.fit(X_train, y_train)

        # Evaluación
        y_pred = self.model.predict(X_test)
        print("\n📊 Reporte de clasificación:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.enc_target.classes_,
            zero_division=0   # ⚡ evita warnings
        ))

        if validar:
            scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")
            print(f"✅ Precisión media (CV 5-fold): {np.mean(scores):.2%}")

    def predecir(self, hora: int, dia_semana: int, provincia: str,
                 precip_acum: float, tipo_via: str = "Ruta Nacional",
                 estado_tiempo: str = "Soleado", estado_calzada: str = "Seca") -> str:
        """
        Predice el tipo de accidente más probable.
        """
        try:
            provincia_enc = self.enc_provincia.transform([provincia])[0]
        except ValueError:
            provincia_enc = -1

        # Categoría hora
        if hora < 6:
            hora_categoria = "Madrugada"
        elif hora < 12:
            hora_categoria = "Mañana"
        elif hora < 18:
            hora_categoria = "Tarde"
        else:
            hora_categoria = "Noche"

        # Categoría lluvia
        if precip_acum < 1:
            lluvia_categoria = "Sin lluvia"
        elif precip_acum < 5:
            lluvia_categoria = "Lluvia leve"
        elif precip_acum < 20:
            lluvia_categoria = "Lluvia moderada"
        else:
            lluvia_categoria = "Lluvia fuerte"

        # Construcción del DataFrame de predicción
        X_nuevo = pd.DataFrame([{
            "hora": hora,
            "dia_semana": dia_semana,
            "mes": 1,
            "fin_de_semana": int(dia_semana in [5, 6]),
            "precip_acum": precip_acum,
            "provincia": provincia_enc,
            "tipo_via": tipo_via,
            "estado del tiempo": estado_tiempo,
            "estado de la calzada": estado_calzada,
            "hora_categoria": hora_categoria,
            "lluvia_categoria": lluvia_categoria
        }])

        for col in ["tipo_via", "estado del tiempo", "estado de la calzada",
                    "hora_categoria", "lluvia_categoria"]:
            if col in X_nuevo.columns:
                X_nuevo[col] = X_nuevo[col].astype("category").cat.codes

        X_nuevo = X_nuevo.fillna(-1)

        # Predicción final
        pred = self.model.predict(X_nuevo)[0]
        return self.enc_target.inverse_transform([pred])[0]

    def probabilidades(self, hora: int, dia_semana: int, provincia: str,
                       precip_acum: float, tipo_via: str = "Ruta Nacional",
                       estado_tiempo: str = "Soleado", estado_calzada: str = "Seca") -> pd.DataFrame:
        """
        Devuelve las probabilidades de cada tipo de accidente en un DataFrame.
        """
        try:
            provincia_enc = self.enc_provincia.transform([provincia])[0]
        except ValueError:
            provincia_enc = -1

        # Categoría hora
        if hora < 6:
            hora_categoria = "Madrugada"
        elif hora < 12:
            hora_categoria = "Mañana"
        elif hora < 18:
            hora_categoria = "Tarde"
        else:
            hora_categoria = "Noche"

        # Categoría lluvia
        if precip_acum < 1:
            lluvia_categoria = "Sin lluvia"
        elif precip_acum < 5:
            lluvia_categoria = "Lluvia leve"
        elif precip_acum < 20:
            lluvia_categoria = "Lluvia moderada"
        else:
            lluvia_categoria = "Lluvia fuerte"

        # Construcción del DataFrame de predicción
        X_nuevo = pd.DataFrame([{
            "hora": hora,
            "dia_semana": dia_semana,
            "mes": 1,
            "fin_de_semana": int(dia_semana in [5, 6]),
            "precip_acum": precip_acum,
            "provincia": provincia_enc,
            "tipo_via": tipo_via,
            "estado del tiempo": estado_tiempo,
            "estado de la calzada": estado_calzada,
            "hora_categoria": hora_categoria,
            "lluvia_categoria": lluvia_categoria
        }])

        for col in ["tipo_via", "estado del tiempo", "estado de la calzada",
                    "hora_categoria", "lluvia_categoria"]:
            if col in X_nuevo.columns:
                X_nuevo[col] = X_nuevo[col].astype("category").cat.codes

        X_nuevo = X_nuevo.fillna(-1)

        # Probabilidades
        probs = self.model.predict_proba(X_nuevo)[0]
        clases = self.enc_target.classes_

        return pd.DataFrame({
            "Tipo de accidente": clases,
            "Probabilidad": probs
        })


# ⚡ Modelo rápido (para pruebas, acepta NaN nativamente)
class ModeloAccidentesRapido(BaseModeloAccidentes):
    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=100,
            random_state=42
        )


# 🏆 Modelo completo (para entrenamiento final)
class ModeloAccidentesCompleto(BaseModeloAccidentes):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
