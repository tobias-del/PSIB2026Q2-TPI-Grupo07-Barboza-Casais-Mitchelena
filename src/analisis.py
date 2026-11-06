"""Lógica de la interfaz gráfica: analiza un corte con el pipeline del TP. No depende de Qt.

Une las funciones ya validadas en los notebooks 02 a 06 (preprocesamiento, segmentación, características,
regla de clasificación por umbrales y compresión) en una sola llamada, `analizar_corte`.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.datos import DATA
from src.compresion import comprimir_jpeg, comprimir_wavelet, calidad_imagen, preprocesar_img
from src.segmentacion import umbral_adaptativo
from src.caracteristicas import medidas_corte, glcm_features, region_profunda, UMBRAL_FRAC
from src.clasificacion_umbrales import REGLA, preparar, ajustar_umbrales

# Umbrales finales del notebook 05 (percentil 95 de los controles de cada eje, calculado con todos los cortes).
# Están redondeados a 4 decimales; si existe el CSV del notebook 04 se recalculan sin redondear.
UMBRALES_POR_DEFECTO = {
    ("Axial", "Area_rel"): 0.0070, ("Axial", "Entropia"): 6.0005,
    ("Sagital", "Area_rel"): 0.0110, ("Sagital", "Entropia"): 6.0083,
}
CSV_CARACTERISTICAS = DATA.parent.parent / "caracteristicas_cortes.csv"   # data/caracteristicas_cortes.csv

METODOS_COMPRESION = ("Ninguna", "JPEG", "Wavelet")
NOMBRES = {"Area_rel": "Área relativa de lesión", "Entropia": "Entropía (textura)"}


def cargar_umbrales(csv=CSV_CARACTERISTICAS):
    """Devuelve (umbrales, origen). Los umbrales son {(eje, característica): valor} de la regla del notebook 05.
    Si existe `csv` se recalculan con `ajustar_umbrales`; si no (o si falla), se usan los valores por defecto."""
    if Path(csv).exists():
        try:
            return ajustar_umbrales(preparar(pd.read_csv(csv)), REGLA), "recalculados con data/caracteristicas_cortes.csv"
        except Exception:
            pass
    return dict(UMBRALES_POR_DEFECTO), "valores del notebook 05 (redondeados a 4 decimales)"


def adivinar_eje(ruta):
    """Eje probable según el nombre del archivo o de su carpeta: 'Sagital' si contiene 'sag' o el nombre empieza
    con 'S' y un número (como 'S1 (12)'); si no, 'Axial'. Es solo la opción inicial: el usuario puede cambiarla."""
    ruta = Path(ruta)
    nombre = ruta.stem.lower()
    if "sag" in str(ruta).lower() or (nombre[:1] == "s" and nombre[1:2].isdigit()):
        return "Sagital"
    return "Axial"


def _comprimir(img, metodo, nivel):
    """Aplica la compresión elegida a la imagen original. `nivel` es la calidad (1-100) en JPEG y el porcentaje de
    coeficientes conservados (0-100) en wavelet. Devuelve (imagen reconstruida, calidad o None)."""
    if metodo == "JPEG":
        rec, _ = comprimir_jpeg(img, int(nivel))
    elif metodo == "Wavelet":
        rec, _ = comprimir_wavelet(img, float(nivel) / 100)
    else:
        return img, None
    return rec, calidad_imagen(img, rec)


def analizar_corte(img, eje, umbrales, metodo="Ninguna", nivel=None):
    """Corre el pipeline completo sobre un corte (imagen 2D uint8) y aplica la regla del notebook 05.

    Pasos: compresión opcional -> preprocesar -> segmentar lesiones -> medidas y textura -> descarte por poco
    cerebro (mismo criterio que el notebook 04) -> regla "Area_rel O Entropía superan el umbral de su eje".
    Con compresión, la regla usa los umbrales fijados con imágenes originales, igual que en el notebook 06.

    Devuelve un dict con las imágenes (`procesada`, `z`, `mascara`, `segmentacion`), `medidas`, `textura`,
    `calidad` (PSNR y SSIM, o None sin compresión), `umbral_frac`, `criterios` (una fila por característica de
    la regla) y `veredicto`: 'descartado', 'marcado' o 'sin_hallazgos'.
    """
    procesada, calidad = _comprimir(img, metodo, nivel)
    z, mascara = preprocesar_img(procesada)
    seg = umbral_adaptativo(z, mascara)
    medidas = medidas_corte(mascara, seg)
    textura = glcm_features(z, region_profunda(mascara, seg))
    valores = {**medidas, **textura}

    umbral_frac = UMBRAL_FRAC[eje]
    criterios = []
    for car, sentido in REGLA:
        valor, umbral = valores[car], umbrales[(eje, car)]
        evaluable = not np.isnan(valor)
        criterios.append({"Caracteristica": car, "Valor": valor, "Umbral": umbral,
                          "Evaluable": evaluable, "Supera": bool(evaluable and valor > umbral)})

    if medidas["Frac_cerebro"] < umbral_frac:
        veredicto = "descartado"
    elif any(c["Supera"] for c in criterios):
        veredicto = "marcado"
    else:
        veredicto = "sin_hallazgos"

    return {"eje": eje, "procesada": procesada, "z": z, "mascara": mascara, "segmentacion": seg,
            "medidas": medidas, "textura": textura, "calidad": calidad, "umbral_frac": umbral_frac,
            "criterios": criterios, "veredicto": veredicto}
