# src/helpers/utilidades.py
from typing import Optional, Iterable
import logging, os

class Utilidades:
    @staticmethod
    def configurar_logs(nivel=logging.INFO):
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=nivel
        )
        return logging.getLogger("accidentes_cr")

    @staticmethod
    def asegurar_directorios(rutas: Iterable[str]):
        for r in rutas:
            os.makedirs(r, exist_ok=True)

    @staticmethod
    def normalizar_texto(s: Optional[str]) -> Optional[str]:
        return s.strip().title() if isinstance(s, str) else s
