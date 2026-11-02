import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure
from skimage.transform import resize
from src.datos import a_gris 

TAM = 256   # lado del cuadrado final (px)


def suavizar(img, tam=3):
    """Filtro de mediana tam x tam."""
    return ndi.median_filter(img.astype(float), size=tam)


def mascara_cerebral(img, radio=6, frac_area=0.25, solidez_min=0.3):
    """Máscara booleana del cerebro: Otsu, erosión, islas compactas y grandes, relleno y dilatación.
    Devuelve una máscara vacía si ninguna isla es compacta (corte sin cerebro útil)."""
    s = suavizar(img)
    mask = s > filters.threshold_otsu(s)
    estructura = morphology.disk(radio)
    er = ndi.binary_erosion(mask, structure=estructura)
    lab = measure.label(er)
    compactas = [p for p in measure.regionprops(lab) if p.solidity >= solidez_min]
    if not compactas:
        return np.zeros_like(mask)
    amax = max(p.area for p in compactas)
    conservar = [p.label for p in compactas if p.area >= frac_area * amax]
    m = ndi.binary_fill_holes(np.isin(lab, conservar))
    return ndi.binary_dilation(m, structure=estructura)


def normalizar(img, mask):
    """Z-score: (x - mediana) / IQR, con mediana e IQR calculados solo dentro de la máscara.
    Solo tiene sentido interpretar los valores dentro de la máscara."""
    if not mask.any():
        return np.zeros(img.shape)
    v = img[mask].astype(float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = (q3 - q1) or 1.0
    return (img.astype(float) - med) / iqr


def a_tamano_comun(arr, tam=TAM, orden=1):
    """Escala manteniendo la proporción (el lado mayor queda en `tam`) y centra en un cuadrado tam x tam, rellenando con ceros."""
    h, w = arr.shape
    f = tam / max(h, w)
    nh, nw = round(h * f), round(w * f)
    chico = resize(arr.astype(float), (nh, nw), order=orden, anti_aliasing=(orden > 0), preserve_range=True)
    out = np.zeros((tam, tam))
    y0, x0 = (tam - nh) // 2, (tam - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = chico
    return out


def preprocesar(ruta):
    """Pipeline completo de preprocesamiento. Devuelve (imagen normalizada TAMxTAM, máscara TAMxTAM)."""
    img = a_gris(ruta)
    m = mascara_cerebral(img)
    z = a_tamano_comun(np.where(m, normalizar(img, m), 0))
    return z, a_tamano_comun(m, orden=0) > 0.5