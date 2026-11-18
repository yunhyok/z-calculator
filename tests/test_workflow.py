import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog
from pptx import Presentation

from zcalc.ui import Window, configure_style


class DesktopWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_style(cls.app)

    def calculate(self):
        self.window.calculate()
        deadline = time.monotonic() + 30
        while self.window.job is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(self.window.job)
        self.assertTrue(self.window.report_button.isEnabled(), self.window.status.text())

    def test_import_synthesize_plot_export(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            settings = QSettings(str(folder / "settings.ini"), QSettings.Format.IniFormat)
            with patch("zcalc.ui.QSettings", return_value=settings):
                self.window = Window()
            self.addCleanup(self.window.close)
            cmx = folder / "sample.cmx"
            cmx.write_text('<Cadence_Material_Lib><Material name="Demo"><Dielectric><Model>'
                           '100 4.3 0.015\n10000 4 0.02</Model></Dielectric></Material></Cadence_Material_Lib>')
            self.window.open_library(cmx)
            self.window.kind.setCurrentIndex(self.window.kind.findData("coupled_stripline"))
            self.window.shield.setChecked(True)
            self.window.bottom.selector.setCurrentIndex(1)
            self.window.fields["angle_deg"].setText("75")
            self.window.mode.setCurrentIndex(1)
            self.calculate()
            snap = self.window.snapshot
            design = list(snap.freq_hz).index(snap.design_hz)
            self.assertEqual(snap.bottom_name, "Demo")
            self.assertEqual(snap.reference_ohm, 100)
            self.assertAlmostEqual(snap.result.impedance_ohm[design], 100, delta=.21)
            self.assertAlmostEqual(float(self.window.fields["width_um"].text()), snap.geometry.width_um, places=4)
            for i in range(self.window.metric_selector.count()):
                self.window.metric_selector.setCurrentIndex(i)
                path = folder / (self.window.metric_selector.currentData() + ".png")
                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")):
                    self.window.export(False)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            path = folder / "report.pptx"
            with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")):
                self.window.export(True)
            slides = Presentation(path).slides
            self.assertGreaterEqual(len(slides), 6)
            self.assertTrue(any("목표 100 Ω" in shape.text for shape in slides[0].shapes if shape.has_text_frame))
            self.window.fields["start"].setText("50")
            self.assertFalse(self.window.report_button.isEnabled())
            self.window.calculate()
            self.assertIsNone(self.window.job)
            self.assertIn("주파수", self.window.status.text())
