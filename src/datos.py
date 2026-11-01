from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image

DATA = Path(__file__).resolve().parent.parent / "data" / "multiple-sclerosis" / "MS"
CARPETAS = {
    "Control Axial_crop":    ("Control", "Axial"),
    "Control Saggital_crop": ("Control", "Sagital"),
    "MS Axial_crop":         ("MS", "Axial"),
    "MS Saggital_crop":      ("MS", "Sagital"),
}
PATRON = re.compile(r"(\d*)\s*[a-zA-Z]?\s*\((\d+)\)\s*$")


def a_gris(ruta):
    """Lee una imagen y la devuelve como matriz 2D de niveles de gris (0-255)."""
    return np.asarray(Image.open(ruta).convert("L"))


def cargar_inventario():
    """DataFrame con una fila por imagen: ruta, Clase, Corte, N (número de corte) e IdNombre."""
    filas = [{"ruta": p, "Clase": c, "Corte": k}
             for carpeta, (c, k) in CARPETAS.items()
             for p in sorted((DATA / carpeta).iterdir())]
    df = pd.DataFrame(filas)
    m = df["ruta"].apply(lambda p: PATRON.search(p.stem))
    df["N"] = m.apply(lambda x: int(x.group(2)))
    df["IdNombre"] = m.apply(lambda x: x.group(1) if x.group(1) else None)
    df.loc[df["Clase"] == "MS", "IdNombre"] = None   # en MS el "1" de A1/S1 es fijo
    return df