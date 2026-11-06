import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Regla elegida en el notebook 05 (clasificación por umbrales, sin aprendizaje automático):
# un corte es EM si alguna de las características supera el percentil P_CONTROL de los controles de su eje.
REGLA = [("Area_rel", ">"), ("Entropia", ">")]
P_CONTROL = 95          # percentil de los controles: fija una tasa de falsos positivos de ~5 % por característica
SEG = ["Area_rel", "N_regiones", "Area_max_region_px"]
TEXT = ["Contraste", "Homogeneidad", "Energia", "Correlacion", "Entropia"]

N_PACIENTES_MS = 72    # pacientes de MS según el dataset
BLOQUE_CONTROL = 14    # archivos por grupo en los controles sin ID (149 cortes / 11 pacientes)
SEMILLA = 0
K = 5


def grupos_aproximados(df, n_pac_ms=N_PACIENTES_MS, bloque_control=BLOQUE_CONTROL, inventario=None):
    """Grupo de "paciente" de cada corte. Control con ID: el ID real. Control sin ID: bloques de
    `bloque_control` archivos consecutivos en cada serie ("kontrol (n)" y "kontrol aks (n)").
    MS: bloques de archivos consecutivos del contador, de tamaño (cortes del eje en `inventario`) / n_pac_ms.
    Es una APROXIMACIÓN: no se sabe si el contador de MS sigue el orden de los pacientes."""
    inv = df if inventario is None else inventario
    g = pd.Series(index=df.index, dtype=object)
    ctrl = df["Clase"] == "Control"
    con_id = ctrl & df["IdNombre"].notna()
    g[con_id] = "C" + df.loc[con_id, "IdNombre"].astype(int).astype(str)
    sin_id = ctrl & df["IdNombre"].isna()
    serie = df.loc[sin_id, "Ruta"].str.contains("aks").map({True: "aks", False: "k"})
    g[sin_id] = "C_" + serie + "_" + ((df.loc[sin_id, "Corte"] - 1) // bloque_control).astype(str)
    for eje in ["Axial", "Sagital"]:
        ms = (df["Clase"] == "MS") & (df["Eje"] == eje)
        n_total = ((inv["Clase"] == "MS") & (inv["Eje"] == eje)).sum()
        tam = round(n_total / n_pac_ms)
        g[ms] = f"MS_{eje[0]}_" + ((df.loc[ms, "Corte"] - 1) // tam).astype(str)
    return g


def preparar(feat):
    """Cortes que pasaron el descarte y sin NaN en las características, con y (1 = MS) y Grupo."""
    datos = feat[feat["Pasa"]].dropna(subset=SEG + TEXT).reset_index(drop=True)
    datos["y"] = (datos["Clase"] == "MS").astype(int)
    datos["Grupo"] = grupos_aproximados(datos, inventario=feat)
    return datos


def particion_por_grupo(grupos, y, k=K, semilla=SEMILLA):
    """Asigna cada grupo a una de k particiones, por separado para cada clase (estratificado) y al azar.
    Devuelve el número de partición de cada corte. Todos los cortes de un grupo quedan juntos."""
    clase = pd.Series(y, index=grupos).groupby(level=0).first()
    rng = np.random.default_rng(semilla)
    particion = {}
    for c in sorted(clase.unique()):
        gs = np.array(sorted(clase[clase == c].index))
        rng.shuffle(gs)
        particion.update({g: i % k for i, g in enumerate(gs)})
    return pd.Series(grupos).map(particion).to_numpy()


def umbral_percentil(valores_control, sentido=">", p=P_CONTROL):
    """Umbral a partir de los controles: percentil p si EM es "mayor" (>), percentil 100 - p si es "menor" (<)."""
    return np.percentile(valores_control, p if sentido == ">" else 100 - p)


def umbral_youden(valores, y, sentido=">"):
    """Umbral que maximiza sensibilidad + especificidad - 1 (índice de Youden), probando los percentiles 1 a 99."""
    mejor, umbral = -np.inf, None
    for u in np.percentile(valores, np.arange(1, 100)):
        pred = valores > u if sentido == ">" else valores < u
        j = pred[y == 1].mean() - pred[y == 0].mean()
        if j > mejor:
            mejor, umbral = j, u
    return umbral


def ajustar_umbrales(df, regla=REGLA, criterio="percentil", p=P_CONTROL):
    """Umbral de cada característica de la regla, por eje, calculado con los cortes de df.
    Devuelve {(eje, característica): umbral}."""
    umbrales = {}
    for eje in ["Axial", "Sagital"]:
        d = df[df["Eje"] == eje]
        for c, s in regla:
            if criterio == "percentil":
                umbrales[(eje, c)] = umbral_percentil(d.loc[d["y"] == 0, c].to_numpy(), s, p)
            else:
                umbrales[(eje, c)] = umbral_youden(d[c].to_numpy(), d["y"].to_numpy(), s)
    return umbrales


def aplicar_regla(df, umbrales, regla=REGLA, combinar="o"):
    """True (EM) si alguna ("o") o todas ("y") las características superan su umbral del eje del corte."""
    votos = []
    for c, s in regla:
        u = np.array([umbrales[(e, c)] for e in df["Eje"]])
        votos.append(df[c].to_numpy() > u if s == ">" else df[c].to_numpy() < u)
    votos = np.vstack(votos)
    return votos.any(axis=0) if combinar == "o" else votos.all(axis=0)


def validar(df, particion, regla=REGLA, criterio="percentil", combinar="o", p=P_CONTROL):
    """Validación cruzada: los umbrales se calculan con las particiones de entrenamiento y la regla se
    aplica a la de prueba. Devuelve la predicción (True = EM) de cada corte."""
    pred = np.zeros(len(df), bool)
    for k in np.unique(particion):
        te = particion == k
        u = ajustar_umbrales(df[~te], regla, criterio, p)
        pred[te] = aplicar_regla(df[te], u, regla, combinar)
    return pred


def auc(y, puntaje):
    """AUC = probabilidad de que un EM al azar tenga mayor puntaje que un control al azar (empates = 1/2).
    Es el U de Mann-Whitney dividido n_EM * n_Control (el CLES del notebook 04)."""
    y = np.asarray(y)
    u, _ = mannwhitneyu(puntaje[y == 1], puntaje[y == 0], alternative="two-sided")
    return u / ((y == 1).sum() * (y == 0).sum())


def metricas(y, pred):
    """Sensibilidad, especificidad, exactitud balanceada y matriz de confusión (EM = positivo)."""
    y, pred = np.asarray(y), np.asarray(pred, bool)
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    sens, esp = tp / (tp + fn), tn / (tn + fp)
    return {"Sensibilidad": sens, "Especificidad": esp, "Exactitud balanceada": (sens + esp) / 2,
            "TP": tp, "FN": fn, "TN": tn, "FP": fp}
