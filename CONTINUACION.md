# Guía de continuación del TPI (Grupo 07)

**Estado actual:** notebooks 01 (exploración), 02 (preprocesamiento) y 03 (segmentación) terminados y pusheados. Faltan el **notebook 04** (características y estadística), el clasificador (posible interfaz), la evaluación de compresión y el informe.

---

## 1. Estilo del repo (respetarlo en todos los notebooks)

Cada notebook cuenta una historia en este orden, y se repite para cada sección:

1. **Subtítulo `##`** que describe qué se va a hacer en esa sección.
2. **Bloques de código** que muestran el desarrollo *experimental*: lo que se probó, incluso lo que no funcionó. Las pruebas fallidas **no se borran ni se modifican** queda el registro de los intentos. Si algo se prueba de nuevo con otros parámetros, va en una celda nueva debajo.
3. **Markdown de cierre** con un resumen breve, que tenga:
   - qué se hizo y con qué parámetros,
   - la **conclusión** a la que se llegó (qué se elige y por qué),
   - las **limitaciones** (qué no cubre, qué supuestos se hicieron).

Reglas generales:
- Una celda hace una cosa. No mandar bloques gigantes de código.
- Los números del markdown tienen que salir de la salida real de las celdas (no inventar ni redondear a ojo).
- Columnas del DataFrame con mayúscula inicial (`Clase`, `Corte`, `Eje`, etc.).
- La carpeta se llama `Notebooks/` (con mayúscula, a propósito).
- Primera celda de cada notebook: `%load_ext autoreload` / `%autoreload 2` e imports desde `src`.

## 2. Estilo del código en `src/`

`src/` contiene **solo las funciones que usa el pipeline final**, ya validadas en un notebook. Los métodos descartados (region growing, textura + k-means) quedan documentados en el notebook 03 y **no** se pasan a `src/`.

Reglas:
- Los parámetros elegidos son constantes en mayúscula al principio del archivo y valores por defecto de la función.
- Cada función lleva un docstring que explica qué hace y qué supone.
- Antes de pasar una función a `src/`, se verifica que da **exactamente** el mismo resultado que la del notebook (`np.array_equal` sobre las imágenes de prueba).
- El notebook importa de `src`, no redefine.

Archivos actuales:
| Archivo | Contenido |
|---|---|
| `src/datos.py` | `cargar_inventario()`, `a_gris()`; arma el DataFrame `Ruta, Clase, Corte, N, IdNombre` |
| `src/preproceso.py` | `suavizar` (mediana 3×3), `mascara_cerebral`, `normalizar` (z-score robusto dentro de la máscara), `a_tamano_comun` (256×256 con padding), `preprocesar` |
| `src/segmentacion.py` | `umbral_adaptativo(z, mask)`: umbral adaptativo + apertura (σ=15, offset=1, z_min=1, apertura radio 2, área mín. 50, erosión 3) |

Archivos a crear en esta siguiente etapa: `src/caracteristicas.py` y, más adelante, `src/clasificacion.py` y `src/compresion.py`.

Al final de cada notebook se agrega un markdown **"Resumen para `src/`"** que lista qué funciones pasaron a `src/`, con qué parámetros y qué quedó afuera.

## 3. Resumen de lo decidido hasta ahora (para el informe)

- **Datos:** Kaggle Multiple Sclerosis Dataset (Macin et al. 2022), FLAIR, 72 MS + 59 controles, cortes axiales y sagitales en PNG. **No hay ground truth de lesiones.** Los archivos de MS (`A1 (n)`, `S1 (n)`) tienen un contador continuo y **no hay ID de paciente**; en Control hay ID en parte de los archivos (`kontrol <ID> (<corte>)`) y 149 axiales sin ID.
- **Preprocesamiento:** mediana 3×3; máscara cerebral por Otsu + erosión (radio 6), filtro de islas por solidez (≥0,3) y área (≥0,25× la mayor), relleno de huecos y dilatación; z-score robusto (mediana/IQR dentro de la máscara); redimensionado a 256×256 preservando proporción.
- **Segmentación:** umbral adaptativo + apertura fue el mejor método de los probados (evaluación cualitativa sobre 6 cortes de MS). Region growing es conservador; textura GLCM + k-means sobresegmenta y no aísla lesiones (los clusters se separaron por el signo del cluster shade, no por brillo).
- **Limitaciones ya detectadas:** sin ground truth; tejido nasal en la máscara de cortes de base de cráneo; bordes periventriculares dan falsos positivos y no se distinguen de lesiones periventriculares reales; la apertura puede borrar lesiones muy chicas; las lesiones yuxtacorticales no están cubiertas por la variante de región profunda; MS tiene ~9 cortes por paciente y Control ~15–24.

---

## 4. Notebook 04: `Notebooks/04_caracteristicas.ipynb`

**Objetivo:** extraer características de **todos** los cortes con el pipeline completo, compararlas entre MS y Control y dejar una tabla lista para el clasificador.

### 4.1 Extraer las medidas básicas de todos los cortes
Subtítulo sugerido: `## 4.1 Extracción de medidas por corte`

- Recorrer `cargar_inventario()` y, para cada imagen, aplicar `preprocesar` → `umbral_adaptativo`.
- Calcular por corte:
  - `Clase`, `Eje` (Axial/Sagital), `Corte`, `IdNombre`, `Ruta`;
  - `Area_cerebro` (píxeles de la máscara) y `Frac_cerebro` (= `Area_cerebro` / 256²);
  - `N_regiones` (componentes conexas de la segmentación, `connectivity=2`);
  - `Area_lesion` y `Area_rel` (= `Area_lesion` / `Area_cerebro`);
  - `Area_media_region` y `Area_max_region`.
- **Guardar en `data/caracteristicas_cortes.csv`** (o `Resultados/`) para no recalcular. Agregar una celda con "si existe el CSV, cargarlo; si no, calcularlo". Considerar que el cálculo tarda: probar primero con una muestra chica.
- **Comprobación:** que no falte ningún corte y que la cantidad de filas coincida con el inventario.

Cierre en markdown: cuántos cortes se procesaron, tiempo, errores si hubo.

### 4.2 Criterio de descarte de cortes con poco cerebro
Subtítulo sugerido: `## 4.2 Descarte de cortes con poco cerebro`

Motivo: los cortes de los extremos (base de cráneo, vértice) tienen muy poco cerebro, no son comparables y generan falsos positivos.

- Graficar el histograma de `Frac_cerebro` **separado por Eje** (axial y sagital) y por Clase.
- Elegir un umbral de descarte a partir de la distribución (por ejemplo, el valle entre la cola de cortes chicos y el grueso), **uno por eje**. Justificarlo con la figura.
- Reportar en una tabla cuántos cortes quedan y se descartan **por Clase y por Eje**.
- Mirar una muestra de cortes descartados y de cortes en el límite para confirmar visualmente que el umbral tiene sentido.
- Esta decisión **no se tiene que hacer distinta por clase** (si no, se introduce sesgo): el mismo umbral para MS y Control.

Cierre: umbral elegido, tabla de conteos, limitaciones (el umbral es heurístico; la proporción de descartados puede diferir entre clases).

### 4.3 Estimar el error de la máscara
Subtítulo sugerido: `## 4.3 Verificación de la máscara cerebral`

- Tomar ~50 cortes al azar (semilla fija), mirar la máscara superpuesta y anotar a mano cuántas son correctas/incorrectas (tejido nasal, anillo de cuero cabelludo, hemisferio perdido).
- Reportar la tasa aproximada de error como limitación en el informe.

### 4.4 Comparación MS vs Control (estadística)
Subtítulo sugerido: `## 4.4 Comparación entre MS y Control`

- Boxplots / violines de `Area_rel` y `N_regiones`, **por Eje**, solo con cortes que pasaron el descarte.
- **Mann-Whitney U** entre MS y Control para cada característica y cada eje. Reportar U, p y tamaño de efecto (por ejemplo, la diferencia de medianas o el *common-language effect size*).
- **Control negativo:** los cortes de Control no deberían tener lesiones; las áreas que aparecen ahí son la tasa de falsos positivos del método. Reportarla explícitamente (mediana y percentiles).
- **Advertencia obligatoria en el markdown:** los cortes de un mismo paciente **no son independientes**, y en MS no hay ID de paciente. Por eso los p-valores están sobreestimados en significancia (pseudorreplicación). Mencionar que es la limitación principal de la inferencia.
- Opcional: agregar por paciente en Control (donde hay ID) para mostrar cómo cambia el resultado.

Cierre: qué características separan las clases, con qué tamaño de efecto, y la advertencia anterior.

### 4.5 Características de textura (GLCM) sobre sustancia blanca
Subtítulo sugerido: `## 4.5 Textura de la sustancia blanca`

Recién después de que 4.1–4.4 funcionen.
- Definir la región de análisis (por ejemplo, la región profunda de la máscara: `distance_transform_edt(mask) > 10`, o la máscara sin la segmentación de lesiones).
- Calcular GLCM (distancia 1, 4 ángulos, 16 o 32 niveles) sobre esa región y extraer contraste, homogeneidad, energía, correlación y entropía **por corte**.
- Agregarlas al CSV y repetir el análisis estadístico de 4.4.
- Recordar: la textura **no sirvió para segmentar**, pero acá se usa como característica global del corte, que es otra pregunta; hay que decir esto en el markdown.

### 4.6 Resumen para `src/`
Subtítulo: `## Resumen para src/`

Crear `src/caracteristicas.py` con las funciones que se usaron (por ejemplo `medidas_corte(z, mask, seg)`, `glcm_features(z, region)`, `extraer_todo(inventario)`), verificar que dan lo mismo que en el notebook, y listar en markdown qué quedó en `src/`, qué decisiones se tomaron (umbral de descarte, región de textura) y qué limitaciones se arrastran.

---

## 5. Después del notebook 04

### Notebook 05: clasificador MS vs sano (`Notebooks/05_clasificacion.ipynb`)
- Entrada: el CSV de características, solo cortes que pasaron el descarte.
- Modelos simples y explicables: regresión logística y, como comparación, árbol de decisión o random forest. Estandarizar las características dentro de cada fold.
- **Validación cruzada**, con la advertencia central: **en MS no hay ID de paciente**, así que no se puede hacer partición por paciente. Opciones, a documentar:
  1. Particionar por paciente solo en Control (donde hay ID) y reportar que en MS el resultado puede estar inflado por fuga entre cortes del mismo paciente.
  2. Reconstruir grupos aproximados de MS (por ejemplo, bloques consecutivos del contador `n`). **Cuidado:** ya se probó reconstruir pacientes por saltos de tamaño y se refutó al validarlo en Control; si se usa bloques consecutivos, aclarar que es una aproximación.
  3. Clasificar a nivel de corte y a nivel de "paciente aproximado" y comparar.
- Métricas: sensibilidad, especificidad, AUC, matriz de confusión. Con clases desbalanceadas por cortes (MS ~9 por paciente, Control ~15–24), usar `class_weight="balanced"` o submuestreo y reportar las dos.
- Importancia de características (coeficientes / permutación) para interpretar.
- Cierre con conclusión y limitaciones honestas: sin ground truth de lesiones, sin partición por paciente en MS, dataset chico.

### Notebook 06: compresión (se dejó para el final a propósito)
- Comprimir con JPEG y con wavelets a varias tasas; medir PSNR y SSIM.
- Evaluar si **la clasificación o la segmentación sobreviven a la compresión** (repetir el pipeline con imágenes comprimidas y comparar `Area_rel`, `N_regiones` y el rendimiento del clasificador). Esa es la pregunta de fondo, no solo la calidad de imagen.
- Resumen para `src/compresion.py`.

### Opcional: interfaz gráfica
Una GUI simple (cargar corte → máscara → segmentación → predicción). Solo si sobra tiempo.

### README
Que tenga: objetivo, estructura del repo, cómo instalar (`requirements.txt`), cómo reproducir (orden de los notebooks), una o dos figuras con resultados clave y las limitaciones.

### Informe (máx. 8 páginas)
Introducción, datos, métodos (preprocesamiento, segmentación con la justificación de la comparación de tres métodos, características, estadística, clasificación, compresión), resultados, discusión con **limitaciones** y conclusiones. Las figuras salen de los notebooks.

## 6. Fechas y chequeos finales
- Entrega: **10/11 a las 17:00** (confirmar con la consigna). Defensas: 12 al 19/11.
- Antes de entregar: reiniciar kernel y correr cada notebook completo de punta a punta (que no dependa de estado viejo); verificar que `src/` importa sin errores; que `data/` está ignorado por git y que el README explica cómo bajar el dataset; que hay commits de los tres integrantes.