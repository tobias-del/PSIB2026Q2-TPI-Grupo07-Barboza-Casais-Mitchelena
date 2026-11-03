import io
import numpy as np
import pandas as pd
import pywt
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from src.preproceso import mascara_cerebral, normalizar, a_tamano_comun
from src.segmentacion import umbral_adaptativo
from src.caracteristicas import medidas_corte, glcm_features, region_profunda

CALIDADES_JPEG = [95, 75, 50, 25, 10, 5]                  # parámetro quality de Pillow
FRACCIONES_WAVELET = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]  # fracción de coeficientes que se conservan
WAVELET = "bior4.4"    # biortogonal 9/7 (CDF), la de JPEG 2000 con pérdida
NIVELES_WAVELET = 3


def comprimir_jpeg(img, calidad):
    """Comprime una imagen uint8 en JPEG con la calidad dada y la decodifica.
    Devuelve (imagen reconstruida uint8, bits por píxel del archivo JPEG)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=calidad)
    datos = buf.getvalue()
    rec = np.asarray(Image.open(io.BytesIO(datos)).convert("L"))
    return rec, 8 * len(datos) / img.size


def comprimir_wavelet(img, fraccion, wavelet=WAVELET, nivel=NIVELES_WAVELET):
    """Transformada wavelet 2D, conserva la `fraccion` de coeficientes de mayor magnitud (umbral duro)
    y reconstruye. La reconstrucción se redondea a uint8, como si se guardara como imagen.
    Devuelve (imagen reconstruida uint8, fracción de coeficientes conservados).
    No incluye la codificación de los coeficientes: la tasa de compresión es nominal (1 / fraccion)."""
    coefs = pywt.wavedec2(img.astype(float), wavelet, level=nivel, mode="periodization")
    arr, cortes = pywt.coeffs_to_array(coefs)
    k = max(1, int(round(fraccion * arr.size)))
    umbral = np.partition(np.abs(arr).ravel(), -k)[-k]
    arr[np.abs(arr) < umbral] = 0
    rec = pywt.waverec2(pywt.array_to_coeffs(arr, cortes, output_format="wavedec2"), wavelet, mode="periodization")
    rec = rec[:img.shape[0], :img.shape[1]]
    return np.clip(np.round(rec), 0, 255).astype(np.uint8), np.count_nonzero(arr) / arr.size


def calidad_imagen(original, comprimida):
    """PSNR (dB) y SSIM entre la imagen original y la comprimida, sobre la imagen completa (rango 0-255)."""
    return {"PSNR": peak_signal_noise_ratio(original, comprimida, data_range=255),
            "SSIM": structural_similarity(original, comprimida, data_range=255)}


def preprocesar_img(img):
    """Igual que src.preproceso.preprocesar, pero recibe la imagen ya leída en vez de la ruta."""
    m = mascara_cerebral(img)
    z = a_tamano_comun(np.where(m, normalizar(img, m), 0))
    return z, a_tamano_comun(m, orden=0) > 0.5


def dice(a, b):
    """Coeficiente de Dice entre dos máscaras booleanas. NaN si las dos están vacías."""
    s = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / s if s else np.nan


def evaluar_corte(img, niveles_jpeg=CALIDADES_JPEG, niveles_wavelet=FRACCIONES_WAVELET):
    """Para una imagen original: corre el pipeline sobre la original y sobre cada nivel de compresión.
    Devuelve una lista de filas con Metodo, Nivel, Tasa (bpp en JPEG, fracción en wavelet), PSNR, SSIM,
    Dice de la máscara y de la segmentación contra la original, y las características del notebook 04."""
    z0, m0 = preprocesar_img(img)
    s0 = umbral_adaptativo(z0, m0)
    filas = [{"Metodo": "Original", "Nivel": "-", "Tasa": np.nan, "PSNR": np.inf, "SSIM": 1.0,
              "Dice_mascara": 1.0, "Dice_seg": dice(s0, s0),
              **medidas_corte(m0, s0), **glcm_features(z0, region_profunda(m0, s0))}]
    variantes = [("JPEG", q, comprimir_jpeg, q) for q in niveles_jpeg] + \
                [("Wavelet", f, comprimir_wavelet, f) for f in niveles_wavelet]
    for metodo, nivel, funcion, param in variantes:
        rec, tasa = funcion(img, param)
        z, m = preprocesar_img(rec)
        s = umbral_adaptativo(z, m)
        filas.append({"Metodo": metodo, "Nivel": nivel, "Tasa": tasa, **calidad_imagen(img, rec),
                      "Dice_mascara": dice(m0, m), "Dice_seg": dice(s0, s),
                      **medidas_corte(m, s), **glcm_features(z, region_profunda(m, s))})
    return filas
