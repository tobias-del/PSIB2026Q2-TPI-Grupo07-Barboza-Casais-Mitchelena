"""Interfaz gráfica del TPI (Repo de Fran (PyQt/PySide)).

Flujo: cargar un corte -> elegir eje y compresión -> Analizar. Muestra el corte original, la máscara cerebral,
las lesiones detectadas y el resultado de la regla del notebook 05. La lógica está en `src/analisis.py`.
Se ejecuta con `python app.py` desde la raíz del repo.
OBS: Las carpetas que dicen control se tratan de pacientes sanos y no se debería detectar nada.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QGroupBox, QPushButton, QLabel, QComboBox, QDoubleSpinBox, QFileDialog,
                               QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox)

from src.datos import a_gris
from src.analisis import (analizar_corte, cargar_umbrales, adivinar_eje, METODOS_COMPRESION, NOMBRES)

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

Z_MIN_VISTA, Z_MAX_VISTA = -3.0, 5.0     # rango de z que se muestra en gris
COLOR_MASCARA = (0, 220, 255)
COLOR_LESION = (255, 40, 40)

VEREDICTOS = {
    "marcado": ("Corte marcado: hallazgos compatibles con EM", "#b3261e"),
    "sin_hallazgos": ("Sin hallazgos por encima de los umbrales", "#2e7d32"),
    "descartado": ("Corte descartado: poco cerebro en el corte", "#5f6368"),
}
AVISO = ("Herramienta de apoyo para el TP, no es un diagnóstico. Por corte, la regla detecta ~53 % de los cortes de "
         "MS (sensibilidad) y acierta ~90 % de los controles (especificidad). Un corte sin hallazgos no descarta EM.")


def _gris_rgb(z, mascara):
    """Imagen z en escala de grises (RGB uint8); lo que está fuera de la máscara queda negro."""
    g = np.clip((z - Z_MIN_VISTA) / (Z_MAX_VISTA - Z_MIN_VISTA), 0, 1)
    g = np.where(mascara, g, 0)
    return np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=-1)


def _borde(m):
    """Píxeles del borde interno de una máscara booleana."""
    return m & ~ndi.binary_erosion(m)


def panel_mascara(z, mascara):
    """Imagen normalizada con el contorno de la máscara cerebral."""
    rgb = _gris_rgb(z, mascara)
    rgb[_borde(mascara)] = COLOR_MASCARA
    return rgb


def panel_lesiones(z, mascara, seg):
    """Imagen normalizada con las lesiones candidatas en rojo (relleno semitransparente y contorno)."""
    rgb = _gris_rgb(z, mascara)
    rgb[seg] = (0.55 * rgb[seg] + 0.45 * np.array(COLOR_LESION)).astype(np.uint8)
    rgb[_borde(seg)] = COLOR_LESION
    return rgb


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TPI Grupo 07 — Detección de lesiones de EM en cortes de RM (FLAIR)")
        self.resize(1280, 780)
        self.umbrales, self.origen_umbrales = cargar_umbrales()
        self.img = None
        self.ruta = None

        central = QWidget()
        self.setCentralWidget(central)
        raiz = QHBoxLayout(central)
        raiz.addWidget(self._crear_controles(), 0)
        raiz.addLayout(self._crear_resultados(), 1)
        self.statusBar().showMessage(f"Umbrales: {self.origen_umbrales}. Cargá un corte para empezar.")

    # ---------- construcción de la ventana ----------
    def _crear_controles(self):
        caja = QWidget()
        caja.setFixedWidth(290)
        v = QVBoxLayout(caja)

        self.boton_cargar = QPushButton("Cargar corte…")
        self.boton_cargar.clicked.connect(self.cargar_imagen)
        self.etiqueta_archivo = QLabel("Ningún archivo cargado")
        self.etiqueta_archivo.setWordWrap(True)
        v.addWidget(self.boton_cargar)
        v.addWidget(self.etiqueta_archivo)

        g_eje = QGroupBox("Corte")
        f = QFormLayout(g_eje)
        self.combo_eje = QComboBox()
        self.combo_eje.addItems(["Axial", "Sagital"])
        f.addRow("Eje:", self.combo_eje)
        v.addWidget(g_eje)

        g_comp = QGroupBox("Compresión previa (opcional)")
        f = QFormLayout(g_comp)
        self.combo_metodo = QComboBox()
        self.combo_metodo.addItems(METODOS_COMPRESION)
        self.combo_metodo.currentTextChanged.connect(self._ajustar_nivel)
        self.spin_nivel = QDoubleSpinBox()
        self.etiqueta_nivel = QLabel("Nivel:")
        f.addRow("Método:", self.combo_metodo)
        f.addRow(self.etiqueta_nivel, self.spin_nivel)
        v.addWidget(g_comp)
        self._ajustar_nivel("Ninguna")

        self.boton_analizar = QPushButton("Analizar")
        self.boton_analizar.setEnabled(False)
        self.boton_analizar.clicked.connect(self.analizar)
        self.boton_guardar = QPushButton("Guardar captura…")
        self.boton_guardar.setEnabled(False)
        self.boton_guardar.clicked.connect(self.guardar_captura)
        v.addWidget(self.boton_analizar)
        v.addWidget(self.boton_guardar)

        aviso = QLabel(AVISO)
        aviso.setWordWrap(True)
        aviso.setStyleSheet("color: #5f6368; font-size: 11px;")
        v.addStretch(1)
        v.addWidget(aviso)
        return caja

    def _crear_resultados(self):
        v = QVBoxLayout()

        self.grafico = pg.GraphicsLayoutWidget()
        self.grafico.setBackground("k")
        self.items, self.plots = [], []
        for col, titulo in enumerate(["Original", "Máscara cerebral", "Lesiones detectadas"]):
            plot = self.grafico.addPlot(row=0, col=col, title=titulo)
            self.plots.append(plot)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.setAspectLocked(True)
            plot.invertY(True)
            plot.setMenuEnabled(False)
            item = pg.ImageItem()
            plot.addItem(item)
            self.items.append(item)
        v.addWidget(self.grafico, 3)

        self.veredicto = QLabel("Sin resultados todavía")
        fuente = QFont()
        fuente.setPointSize(14)
        fuente.setBold(True)
        self.veredicto.setFont(fuente)
        self.veredicto.setAlignment(Qt.AlignCenter)
        self.veredicto.setMinimumHeight(44)
        self.veredicto.setStyleSheet("background: #e0e0e0; color: #202124; border-radius: 6px;")
        v.addWidget(self.veredicto)

        self.tabla = QTableWidget(0, 4)
        self.tabla.setHorizontalHeaderLabels(["Medida", "Valor", "Umbral", "Resultado"])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla.setSelectionMode(QTableWidget.NoSelection)
        v.addWidget(self.tabla, 2)

        self.etiqueta_calidad = QLabel("")
        v.addWidget(self.etiqueta_calidad)
        return v

    def _ajustar_nivel(self, metodo):
        """Adapta el control de nivel al método de compresión elegido."""
        s = self.spin_nivel
        s.setEnabled(metodo != "Ninguna")
        if metodo == "JPEG":
            self.etiqueta_nivel.setText("Calidad (1–100):")
            s.setDecimals(0); s.setRange(1, 100); s.setSingleStep(5); s.setSuffix(""); s.setValue(50)
        elif metodo == "Wavelet":
            self.etiqueta_nivel.setText("Coef. conservados:")
            s.setDecimals(1); s.setRange(0.5, 20); s.setSingleStep(0.5); s.setSuffix(" %"); s.setValue(5)
        else:
            self.etiqueta_nivel.setText("Nivel:")
            s.setSuffix("")

    # ---------- acciones ----------
    def cargar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Elegir un corte", "", "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not ruta:
            return
        try:
            self.img = a_gris(ruta)
        except Exception as e:
            QMessageBox.warning(self, "No se pudo abrir la imagen", str(e))
            return
        self.ruta = Path(ruta)
        self.etiqueta_archivo.setText(f"{self.ruta.name}\n{self.img.shape[1]} × {self.img.shape[0]} px")
        self.combo_eje.setCurrentText(adivinar_eje(ruta))
        self.boton_analizar.setEnabled(True)
        self.analizar()

    def analizar(self):
        if self.img is None:
            return
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            r = analizar_corte(self.img, self.combo_eje.currentText(), self.umbrales,
                               self.combo_metodo.currentText(), self.spin_nivel.value())
        except Exception as e:
            QMessageBox.critical(self, "Error al analizar", f"{type(e).__name__}: {e}")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
        self.mostrar(r)

    def mostrar(self, r):
        self.items[0].setImage(r["procesada"], levels=(0, 255))
        self.items[1].setImage(panel_mascara(r["z"], r["mascara"]))
        self.items[2].setImage(panel_lesiones(r["z"], r["mascara"], r["segmentacion"]))
        for plot in self.plots:
            plot.autoRange()

        texto, color = VEREDICTOS[r["veredicto"]]
        self.veredicto.setText(texto)
        self.veredicto.setStyleSheet(f"background: {color}; color: white; border-radius: 6px;")
        self._llenar_tabla(r)

        q = r["calidad"]
        self.etiqueta_calidad.setText(f"Compresión: PSNR = {q['PSNR']:.1f} dB, SSIM = {q['SSIM']:.3f} (contra el original)"
                                      if q else "Sin compresión previa.")
        self.boton_guardar.setEnabled(True)
        self.statusBar().showMessage(f"Analizado: {self.ruta.name if self.ruta else ''} ({r['eje']})")

    def _llenar_tabla(self, r):
        m, descartado = r["medidas"], r["veredicto"] == "descartado"
        filas = [("Fracción de cerebro en el corte", f"{m['Frac_cerebro']:.3f}", f"≥ {r['umbral_frac']:.2f}",
                  "Descartado" if descartado else "Se analiza")]
        for c in r["criterios"]:
            if descartado:
                res = "No se evalúa"
            elif not c["Evaluable"]:
                res = "No evaluable (región chica)"
            else:
                res = "Supera el umbral" if c["Supera"] else "No supera"
            valor = "—" if not c["Evaluable"] else f"{c['Valor']:.4f}" if c["Caracteristica"] == "Area_rel" else f"{c['Valor']:.3f}"
            filas.append((NOMBRES[c["Caracteristica"]], valor, f"> {c['Umbral']:.4f}", res))
        filas += [("Regiones detectadas", str(m["N_regiones"]), "", ""),
                  ("Área de lesión (px)", str(m["Area_lesion_px"]), "", ""),
                  ("Región más grande (px)", str(m["Area_max_region_px"]), "", "")]
        self.tabla.setRowCount(len(filas))
        for i, fila in enumerate(filas):
            for j, texto in enumerate(fila):
                item = QTableWidgetItem(texto)
                if j == 3 and texto == "Supera el umbral":
                    item.setForeground(pg.mkColor("#b3261e"))
                self.tabla.setItem(i, j, item)

    def guardar_captura(self):
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar captura", "resultado.png", "PNG (*.png)")
        if ruta:
            self.grab().save(ruta)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())
