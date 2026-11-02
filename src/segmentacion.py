import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

SIGMA = 15
OFFSET = 1.0
Z_MIN = 1.0
AREA_MIN = 50
EROSION = 3
APERTURA = 2


def umbral_adaptativo(z, mask, sigma=SIGMA, offset=OFFSET, z_min=Z_MIN,
                      area_min=AREA_MIN, erosion=EROSION, apertura=APERTURA):
    """Segmenta lesiones hiperintensas con umbral adaptativo + apertura.

    Un píxel es candidato si supera la media local (gaussiana, calculada solo
    dentro de la máscara cerebral) en `offset` unidades de z y además z > z_min.
    Luego se aplica apertura morfológica y se descartan regiones de área < area_min.
    """   
    m = mask.astype(float)

    media_local = ndi.gaussian_filter(z * m, sigma) / np.maximum(ndi.gaussian_filter(m, sigma), 1e-6)
    interior = ndi.binary_erosion(mask, structure=morphology.disk(erosion))

    seg = (z > media_local + offset) & (z > z_min) & interior

    if apertura > 0:
        seg = ndi.binary_opening(seg, structure=morphology.disk(apertura))   # saca tiras finas

    return morphology.remove_small_objects(seg, area_min)
   