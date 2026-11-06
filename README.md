# PSIB 2026 2C — TPI Grupo 07

**Detección de lesiones de esclerosis múltiple en resonancia magnética FLAIR y clasificación EM vs. control**

16.63 Procesamiento de Señales e Imágenes Biomédicas — ITBA, 2.º cuatrimestre 2026
Tobías Barboza · Martina Casais · Dolores Mitchelena

---

## Objetivo

Construir un pipeline de procesamiento de imágenes clásico, **sin aprendizaje automático**, que:

1. aísle el cerebro y normalice la intensidad de cortes FLAIR 2D;
2. segmente las lesiones hiperintensas compatibles con esclerosis múltiple (EM);
3. extraiga características de cada corte (carga lesional y textura de la sustancia blanca) y las compare entre pacientes con EM y controles;
4. clasifique los cortes como EM o control con una regla de umbrales;
5. evalúe si la segmentación y la clasificación **sobreviven a la compresión** con pérdida (JPEG y wavelets);
6. reúna todo en una **interfaz gráfica** que analiza un corte a la vez.

## Datos

**Multiple Sclerosis Dataset** de Kaggle (Macin et al., 2022): 3427 cortes FLAIR en PNG, ya recortados, de 131 sujetos (72 con EM y 59 controles), en cortes axiales y sagitales. Las imágenes se adquirieron en 2021 en la Facultad de Medicina de la Universidad Özal (Turquía).

| | Axial | Sagital |
|---|---|---|
| Control | 1002 | 1014 |
| EM | 650 | 761 |

Enlace: <https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis>

**El dataset ya está incluido en el repo**, en `data/multiple-sclerosis/MS/`, así que no hace falta descargarlo. La estructura es:

```
data/
└── multiple-sclerosis/
    └── MS/
        ├── Control Axial_crop/
        ├── Control Saggital_crop/     («Saggital» con doble g, como viene en el dataset)
        ├── MS Axial_crop/
        └── MS Saggital_crop/
```

**Dos características del dataset condicionan todo el análisis:**
- **No hay ground truth de lesiones.** La segmentación solo se puede evaluar de forma cualitativa o por consistencia.
- **En EM no hay ID de paciente.** Los archivos se llaman `A1 (n)` y `S1 (n)`, con un contador continuo. En Control, parte de los archivos sí tiene ID (`kontrol <ID> (<corte>)`).

## Instalación

Requiere **Python 3.11 o superior** (lo exige `scikit-image` 0.26).

```bash
git clone https://github.com/tobias-del/PSIB2026Q2-TPI-Grupo07-Barboza-Casais-Mitchelena.git
cd PSIB2026Q2-TPI-Grupo07-Barboza-Casais-Mitchelena
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Cómo reproducir los resultados

Correr los notebooks **en orden**, desde la carpeta `Notebooks/`, reiniciando el kernel antes de cada uno (*Restart Kernel and Run All*).

| Notebook | Qué hace | Genera | Tiempo aprox. |
|---|---|---|---|
| `01_EXPLORACION` | Inventario, formatos, estructura por paciente, histogramas | — | < 1 min |
| `02_PREPROCESAMIENTO` | Suavizado, máscara cerebral, normalización, tamaño común | — | < 1 min |
| `03_SEGMENTACION` | Comparación de tres métodos de segmentación | — | ~ 2 min |
| `04_CARACTERISTICAS` | Medidas por corte, descarte de cortes, estadística EM vs. control, textura | `data/caracteristicas_cortes.csv` | 15–20 min |
| `05_CLASIFICACION_UMBRALES` | Regla de clasificación por umbrales y su validación | — | < 1 min |
| `06_COMPRESION` | JPEG y wavelets: calidad de imagen y efecto sobre segmentación y clasificación | `data/compresion_muestra.csv` | 10–20 min |

- El 05 y el 06 leen el CSV que genera el 04. El 06 usa además la regla del 05.
- El 04 y el 06 guardan sus resultados en CSV. Si el CSV ya existe, lo cargan en lugar de recalcular. **Para recalcular desde cero, hay que borrar los CSV de `data/`.**
- Las celdas largas (extracción del 04, pipeline sobre imágenes comprimidas del 06) no muestran avance hasta terminar: no hay que interrumpirlas.

## Interfaz gráfica

Desde la raíz del repo:

```bash
python app.py
```

La interfaz (PySide6 + pyqtgraph) aplica el pipeline completo a **un corte a la vez**.

**Uso:**
1. **Cargar corte…**: elegir una imagen (PNG, JPG, BMP o TIFF). Puede ser cualquier corte de `data/multiple-sclerosis/MS/`.
2. **Eje**: se propone automáticamente según el nombre del archivo (`S1 (n)` o «sag» → sagital; el resto → axial). Se puede cambiar a mano, y conviene revisarlo, porque los umbrales dependen del eje.
3. **Compresión previa** (opcional): ninguna, JPEG (calidad de 1 a 100) o wavelet (de 0,5 % a 20 % de coeficientes conservados). Sirve para ver en un caso concreto el efecto que estudia el notebook 06.
4. **Analizar**: se ejecuta automáticamente al cargar el corte, y de nuevo cada vez que se aprieta el botón.
5. **Guardar captura…**: guarda la ventana como PNG.

**Qué muestra:**
- **Tres paneles:** el corte original (o el comprimido), la máscara cerebral (contorno celeste) y las lesiones detectadas (en rojo).
- **Un veredicto:**
  - **«Corte marcado: hallazgos compatibles con EM»:** la fracción de cerebro segmentada como lesión (`Area_rel`) o la entropía de la sustancia blanca superan el umbral del eje;
  - **«Sin hallazgos por encima de los umbrales»:** ninguna de las dos lo supera;
  - **«Corte descartado: poco cerebro en el corte»:** el corte es de un extremo de la cabeza y no se analiza (mismo criterio que el notebook 04).
- **Una tabla** con cada medida, su valor, su umbral y si lo supera; además, la cantidad de regiones detectadas, el área de lesión y el área de la región más grande, en píxeles.
- **PSNR y SSIM** contra la imagen original, si se aplicó compresión.

**Umbrales:** la barra de estado indica de dónde salen. Si existe `data/caracteristicas_cortes.csv` (generado por el notebook 04), se recalculan a partir de él. Si no, se usan los valores del notebook 05 redondeados a 4 decimales. En los dos casos, el resultado prácticamente no cambia.

> **No es una herramienta diagnóstica.** Por corte, la regla detecta alrededor del 53 % de los cortes de EM y acierta alrededor del 90 % de los controles. Un corte sin hallazgos no descarta EM.

## Estructura del repo

```
├── app.py                 # lanza la interfaz gráfica
├── Notebooks/             # desarrollo experimental, en orden (incluye las pruebas descartadas)
├── src/                   # funciones del pipeline final, validadas en los notebooks
│   ├── datos.py                    # inventario del dataset y lectura en grises
│   ├── preproceso.py               # mediana 3×3, máscara cerebral, z-score robusto, 256×256
│   ├── segmentacion.py             # umbral adaptativo + apertura
│   ├── caracteristicas.py          # medidas por corte, descarte, textura GLCM
│   ├── clasificacion_umbrales.py   # grupos de paciente, umbrales, regla de decisión, validación
│   ├── compresion.py               # JPEG, wavelets, PSNR/SSIM, Dice, pipeline sobre imágenes comprimidas
│   ├── analisis.py                 # análisis de un corte para la interfaz (sin dependencia de Qt)
│   └── interfaz.py                 # ventana de la interfaz (PySide6 + pyqtgraph)
├── data/                  # dataset y CSV generados por los notebooks
├── requirements.txt
└── CONTINUACION.md        # guía interna de trabajo del grupo
```

Cada notebook sigue la misma estructura: un subtítulo por sección, celdas con las pruebas (las que no funcionaron no se borran) y un texto de cierre con lo que se hizo, la conclusión y las limitaciones.

## Método

1. **Preprocesamiento** (notebook 02):
   - filtro de mediana 3×3;
   - máscara cerebral: umbral de Otsu, erosión (radio 6), selección de las islas compactas y grandes, relleno de huecos y dilatación;
   - normalización z-score robusta (mediana e IQR calculados dentro de la máscara);
   - redimensionado a 256 × 256 px preservando la proporción.
2. **Segmentación** (notebook 03): **umbral adaptativo**. Un píxel es candidato a lesión si supera en 1 unidad de z la media de su entorno (gaussiana con σ = 15 px, calculada dentro de la máscara) y además tiene z > 1. Después se aplica una apertura de radio 2 y se descartan las regiones de menos de 50 px. Se eligió frente a *region growing* (demasiado conservador) y a textura + k-means (sobresegmenta).
3. **Características** (notebook 04): fracción de cerebro en el corte, número de regiones, área de lesión absoluta (px) y relativa al cerebro (`Area_rel`), área media y máxima de las regiones, y textura de la región profunda del cerebro sin lesiones (GLCM: contraste, homogeneidad, energía, correlación y entropía).
4. **Descarte de cortes** (notebook 04): se excluyen los cortes de los extremos de la cabeza, con fracción de cerebro menor que 0,17 (axial) o 0,13 (sagital). El umbral es el mismo para las dos clases.
5. **Clasificación** (notebook 05), **sin aprendizaje automático**: un corte es EM si `Area_rel` **o** la entropía superan el **percentil 95 de los controles** de su eje. Con ese percentil se acepta de antemano alrededor de un 5 % de falsos positivos por característica. La regla se evalúa con validación cruzada en 5 particiones, manteniendo juntos los cortes de cada paciente (aproximado en EM).
6. **Compresión** (notebook 06): JPEG (calidad 95 a 5) y wavelets biortogonales 9/7 (20 % a 0,5 % de los coeficientes) sobre una muestra de 300 cortes. Se mide la calidad de imagen y se repite el pipeline sobre cada versión comprimida.

## Resultados principales

- **EM vs. control:** todas las medidas de la segmentación son mayores en EM. La probabilidad de que un corte de EM tomado al azar supere a uno de control (CLES) es de ≈ 0,79 en axial y ≈ 0,66 en sagital. La sustancia blanca de EM es además más heterogénea: más contraste y entropía (CLES ≈ 0,70).
- **Falsos positivos:** el método marca alguna región en el 23–29 % de los cortes de control.
- **Clasificación:** ninguna característica separa bien sola (AUC ≤ 0,79). La regla combinada alcanza una **sensibilidad de 0,53 y una especificidad de 0,90** por corte (exactitud balanceada 0,72), mejor en axial (0,76) que en sagital (0,68). Agregando los cortes de cada paciente (aproximado en EM), detecta el 76 % de los grupos de EM con un 5 % de falsos positivos.
- **Compresión:**
  - la segmentación se mantiene estable hasta ≈ 1:24 con JPEG (calidad ≥ 25) y ≈ 1:20 con wavelets;
  - la clasificación, hasta ≈ 1:17 con JPEG (calidad ≥ 50) y ≈ 1:10 con wavelets;
  - con más compresión, la entropía baja (los artefactos aplanan la textura) y la segmentación genera falsos positivos, lo que hace caer la especificidad. Un SSIM alto no garantiza que se conserven las lesiones chicas.

## Limitaciones

- **Sin ground truth de lesiones:** la segmentación se eligió por inspección visual y no se validó contra anotaciones de especialistas. Todas las medidas reflejan lo que marca el método, falsos positivos incluidos.
- **Sin ID de paciente en EM:** los grupos de la validación cruzada son bloques de archivos consecutivos, una aproximación. Los cortes de un mismo paciente no son independientes, así que los p-valores del análisis estadístico están sobreestimados (pseudorreplicación); los tamaños de efecto son más informativos.
- **Falsos positivos del método:** bordes de los ventrículos, tejido extracerebral en cortes de base de cráneo y médula en cortes sagitales mediales. Las lesiones yuxtacorticales y las muy chicas pueden no detectarse.
- **Sensibilidad baja por corte:** casi la mitad de los cortes de EM no se detecta, en parte porque muchos no contienen lesiones visibles.
- **Descarte asimétrico:** se descarta el 30 % y el 24 % de los cortes de control (axial y sagital), contra el 10 % y el 7 % de EM, porque los controles tienen más cortes por paciente.
- **Posibles diferencias de adquisición** entre EM y control, que el dataset no informa y que podrían reflejarse en las características, sobre todo en la textura.
- **Dataset chico** (131 sujetos) y ya procesado (PNG recortados y remuestreados, sin espaciado de píxel).

## Referencias

- Macin, G., Tasci, B., Tasci, I., Faust, O., Barua, P. D., Dogan, S., Tuncer, T., Tan, R.-S. y Acharya, U. R. (2022). An Accurate Multiple Sclerosis Detection Model Based on Exemplar Multiple Parameters Local Phase Quantization: ExMPLPQ. *Applied Sciences*, 12(10), 4920. <https://doi.org/10.3390/app12104920>
- Multiple Sclerosis Dataset (Kaggle). <https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis>