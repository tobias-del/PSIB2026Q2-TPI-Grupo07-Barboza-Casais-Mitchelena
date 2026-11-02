import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import graycomatrix

from src.datos import DATA
from src.preproceso import preprocesar, TAM
from src.segmentacion import umbral_adaptativo

# Descarte de cortes con poco cerebro (notebook 04, sección 4.2): mismo umbral para MS y Control
UMBRAL_FRAC = {"Axial": 0.17, "Sagital": 0.13}

# Textura de la sustancia blanca (notebook 04, sección 4.5)
D_PROFUNDA = 10        # px: la región de análisis queda a más de D_PROFUNDA px del borde de la máscara
NIVELES = 32           # niveles de gris de la GLCM
Z_RANGO = (-2.0, 6.0)  # rango de z que se cuantiza (lo de afuera se satura)
PX_MIN_REGION = 200    # px: si la región profunda es más chica, la textura queda en NaN
ANGULOS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def medidas_corte(mask, seg):
    """Medidas básicas de un corte a partir de la máscara cerebral y la segmentación (ambas TAM x TAM).

    Las regiones se cuentan con conectividad 2 (8 vecinos). Si no hay regiones, el área media y
    la máxima valen 0. Si la máscara está vacía, Frac_cerebro vale 0 y Area_rel queda en NaN.
    """
    area_cerebro = int(mask.sum())
    lab = measure.label(seg, connectivity=2)
    areas = np.bincount(lab.ravel())[1:]
    area_lesion = int(seg.sum())
    return {
        "Area_cerebro": area_cerebro,
        "Frac_cerebro": area_cerebro / TAM ** 2,
        "N_regiones": len(areas),
        "Area_lesion": area_lesion,
        "Area_rel": area_lesion / area_cerebro if area_cerebro else np.nan,
        "Area_media_region": float(areas.mean()) if len(areas) else 0.0,
        "Area_max_region": int(areas.max()) if len(areas) else 0,
    }


def pasa_descarte(df, umbral=UMBRAL_FRAC):
    """Serie booleana: True si Frac_cerebro >= umbral del eje del corte (mismo umbral para MS y Control)."""
    return (df["Frac_cerebro"] >= df["Eje"].map(umbral)).rename("Pasa")


def region_profunda(mask, seg, d_min=D_PROFUNDA):
    """Región de análisis de textura: píxeles a más de d_min px del borde de la máscara y fuera de la
    segmentación de lesiones. Supone que lo que queda es mayormente sustancia blanca (no hay atlas)."""
    return (ndi.distance_transform_edt(mask) > d_min) & ~seg


def glcm_features(z, region, niveles=NIVELES, z_rango=Z_RANGO, px_min=PX_MIN_REGION):
    """Contraste, homogeneidad, energía, correlación y entropía de la GLCM restringida a `region`.

    z se cuantiza en `niveles` niveles dentro de z_rango. Los píxeles fuera de la región se llevan a
    un nivel extra que después se descarta, así solo cuentan los pares de píxeles con ambos extremos
    dentro de la región. GLCM simétrica, distancia 1, acumulada en los 4 ángulos y normalizada.
    Energía = suma de P² (segundo momento angular), igual que en el notebook 03.
    """
    nombres = ["Contraste", "Homogeneidad", "Energia", "Correlacion", "Entropia"]
    if region.sum() < px_min:
        return {n: np.nan for n in nombres}
    zmin, zmax = z_rango
    q = (np.clip((z - zmin) / (zmax - zmin), 0, 0.9999) * niveles).astype(np.uint8)
    q[~region] = niveles
    g = graycomatrix(q, distances=[1], angles=ANGULOS, levels=niveles + 1, symmetric=True)
    P = g[:niveles, :niveles, 0, :].sum(axis=-1).astype(float)
    P /= P.sum()
    i, j = np.indices(P.shape)
    mu_i, mu_j = (P * i).sum(), (P * j).sum()
    sd_i = np.sqrt((P * (i - mu_i) ** 2).sum())
    sd_j = np.sqrt((P * (j - mu_j) ** 2).sum())
    nz = P[P > 0]
    return {
        "Contraste": float((P * (i - j) ** 2).sum()),
        "Homogeneidad": float((P / (1 + (i - j) ** 2)).sum()),
        "Energia": float((P ** 2).sum()),
        "Correlacion": float((P * (i - mu_i) * (j - mu_j)).sum() / (sd_i * sd_j)) if sd_i * sd_j > 0 else np.nan,
        "Entropia": float(-(nz * np.log2(nz)).sum()),
    }


def tabla_base(inventario):
    """Pasa el inventario de cargar_inventario() a las columnas del notebook 04:
    Ruta (relativa a DATA), Clase, Eje (Axial/Sagital), Corte (número del archivo), IdNombre."""
    return pd.DataFrame({
        "Ruta": inventario["ruta"].apply(lambda p: p.relative_to(DATA).as_posix()),
        "Clase": inventario["Clase"],
        "Eje": inventario["Corte"],
        "Corte": inventario["N"],
        "IdNombre": inventario["IdNombre"],
    }).reset_index(drop=True)


def extraer_todo(inventario, textura=True):
    """Aplica preprocesar -> umbral_adaptativo a cada imagen del inventario y devuelve
    (DataFrame con una fila por corte, lista de (Ruta, error) de los cortes que fallaron)."""
    base = tabla_base(inventario)
    filas, errores = [], []
    for ruta_abs, (_, b) in zip(inventario["ruta"], base.iterrows()):
        try:
            z, mask = preprocesar(ruta_abs)
            seg = umbral_adaptativo(z, mask)
            fila = {**b.to_dict(), **medidas_corte(mask, seg)}
            if textura:
                fila.update(glcm_features(z, region_profunda(mask, seg)))
            filas.append(fila)
        except Exception as e:
            errores.append((b["Ruta"], repr(e)))
    return pd.DataFrame(filas), errores
