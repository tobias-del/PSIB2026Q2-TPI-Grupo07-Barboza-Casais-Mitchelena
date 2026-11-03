import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

# Características elegidas en el notebook 05
SEG = ["Area_rel", "N_regiones", "Area_max_region"]
TEXT = ["Contraste", "Homogeneidad", "Energia", "Correlacion", "Entropia"]
CARACTERISTICAS = SEG + TEXT + ["Eje_axial"]

N_PACIENTES_MS = 72    # pacientes de MS según el dataset
BLOQUE_CONTROL = 14    # archivos por grupo en los controles sin ID (149 cortes / 11 pacientes)
SEMILLA = 0
K = 5


def grupos_aproximados(df, n_pac_ms=N_PACIENTES_MS, bloque_control=BLOQUE_CONTROL, inventario=None):
    """Grupo de "paciente" de cada corte, para la validación cruzada por grupos.

    Control con ID: el ID real. Control sin ID: bloques de `bloque_control` archivos consecutivos
    dentro de cada serie ("kontrol (n)" y "kontrol aks (n)"). MS: bloques de archivos consecutivos
    del contador n, de tamaño (cortes del eje en `inventario`) / n_pac_ms. Es una APROXIMACIÓN:
    no se sabe si el contador de MS sigue el orden de los pacientes.
    `inventario` es la tabla completa (sin descarte) de la que se calcula el tamaño de bloque.
    """
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
    """Cortes que pasaron el descarte y sin NaN en las características, con Eje_axial, y (1 = MS) y Grupo."""
    datos = feat[feat["Pasa"]].dropna(subset=SEG + TEXT).reset_index(drop=True)
    datos["Eje_axial"] = (datos["Eje"] == "Axial").astype(int)
    datos["y"] = (datos["Clase"] == "MS").astype(int)
    datos["Grupo"] = grupos_aproximados(datos, inventario=feat)
    return datos


def modelo_logistica(class_weight="balanced"):
    """Estandarización + regresión logística (la estandarización se ajusta en cada partición)."""
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight=class_weight, max_iter=1000))


def modelo_random_forest(class_weight="balanced", semilla=SEMILLA):
    """Estandarización + random forest de 300 árboles con hojas de al menos 5 cortes."""
    return make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight=class_weight, random_state=semilla, n_jobs=1))


def submuestrear(idx, y, rng):
    """Índices de entrenamiento con la clase mayoritaria submuestreada al tamaño de la minoritaria."""
    clases, cuentas = np.unique(y[idx], return_counts=True)
    n = cuentas.min()
    return np.concatenate([rng.choice(idx[y[idx] == c], n, replace=False) for c in clases])


def validar(modelo, X, y, grupos=None, balanceo="class_weight", k=K, semilla=SEMILLA):
    """Validación cruzada estratificada (por grupo si se pasan `grupos`).
    Devuelve las probabilidades de MS fuera de la partición y el AUC de cada partición.
    balanceo="submuestreo" submuestrea la clase mayoritaria en cada entrenamiento."""
    if grupos is None:
        cv = StratifiedKFold(k, shuffle=True, random_state=semilla).split(X, y)
    else:
        cv = StratifiedGroupKFold(k, shuffle=True, random_state=semilla).split(X, y, grupos)
    rng = np.random.default_rng(semilla)
    proba, aucs = np.zeros(len(y)), []
    for tr, te in cv:
        if balanceo == "submuestreo":
            tr = submuestrear(tr, y, rng)
        m = modelo.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba[te]))
    return proba, np.array(aucs)


def metricas(y, proba, umbral=0.5):
    """AUC, sensibilidad y especificidad (MS = positivo) y la matriz de confusión con umbral 0,5."""
    tn, fp, fn, tp = confusion_matrix(y, proba >= umbral).ravel()
    return {"AUC": roc_auc_score(y, proba), "Sensibilidad": tp / (tp + fn), "Especificidad": tn / (tn + fp),
            "TP": tp, "FN": fn, "TN": tn, "FP": fp}
