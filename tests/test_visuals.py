"""Real Qt rendering and file round-trip checks, using synthetic results."""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from pptx import Presentation
from PySide6.QtCore import QSize
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from zcalc.engine import Geometry
from zcalc.visuals import (METRICS, PlotWidget, SectionWidget, Snapshot, _at_design,
                           _x_ticks, _y_ticks, save_png, save_pptx)


def snapshot(kind="coupled_stripline", shield=True):
    f = np.geomspace(1e6, 1e10, 41)
    result = SimpleNamespace(
        impedance_ohm=100 + np.linspace(2, -2, f.size),
        return_loss_db=20 + 3 * np.sin(np.linspace(0, 8, f.size)),
        attenuation_db_m=np.sqrt(f / 1e9),
        insertion_loss_db=.1 * np.sqrt(f / 1e9),
        dcr_ohm_m=2.1,
        warnings=("Cross-section exceeds one tenth wavelength; higher-order/full-wave validation required.",),
    )
    return Snapshot(Geometry(kind=kind, shield=shield, angle_deg=75), result, f,
                    "하부 시험 재료", "상부 시험 재료", np.full(f.shape, 4.2),
                    np.full(f.shape, .02), np.full(f.shape, 3.8), np.full(f.shape, .015),
                    .1, 100, 1e9)


class VisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_snapshot_rejects_invalid_frequencies_and_shapes(self):
        s = snapshot()
        for frequencies in ([0, 1], [1, 1], [2, 1], [1, np.nan], [1]):
            with self.subTest(frequencies=frequencies), self.assertRaises(ValueError):
                replace(s, freq_hz=np.asarray(frequencies))
        with self.assertRaises(ValueError):
            replace(s, bottom_er=np.ones(3))
        with self.assertRaises(ValueError):
            replace(s, reference_ohm=0)

    def test_log_interpolation_and_ticks(self):
        s = replace(snapshot(), design_hz=np.sqrt(1e6 * 1e10))
        self.assertAlmostEqual(_at_design(s, np.log10(s.freq_hz)), 8)
        major, _ = _x_ticks(1e6, 1e10)
        self.assertEqual(major, [1e6, 1e7, 1e8, 1e9, 1e10])
        low, high, ticks = _y_ticks(np.full(20, 2.1))
        self.assertLess(low, 2.1)
        self.assertGreater(high, 2.1)
        self.assertTrue(np.all(np.diff(ticks) > 0))

    def test_all_six_png_exports_and_geometry_variants(self):
        s = snapshot()
        with tempfile.TemporaryDirectory() as folder:
            for key in ["section", *METRICS]:
                with self.subTest(metric=key):
                    output = Path(folder) / f"{key}.png"
                    save_png(output, s, key)
                    picture = QImage(str(output))
                    self.assertFalse(picture.isNull())
                    self.assertEqual(picture.size(), QSize(1760, 992))
                    self.assertGreater(output.stat().st_size, 12000)
            for kind in ("microstrip", "stripline", "coupled_microstrip", "coupled_stripline"):
                with self.subTest(kind=kind):
                    for shield in (False, True):
                        save_png(Path(folder) / "section.png", snapshot(kind, shield), "section")

    def test_compact_widget_and_infinite_return_loss(self):
        s = snapshot()
        s.result.return_loss_db = np.full(s.freq_hz.shape, np.inf)
        for widget in (SectionWidget(), PlotWidget()):
            if isinstance(widget, PlotWidget):
                widget.set_metric("return")
            widget.set_snapshot(s)
            for size in (QSize(550, 210), QSize(820, 245), QSize(1000, 564)):
                widget.resize(size)
                rendered = widget.grab().toImage()
                self.assertFalse(rendered.isNull())
                self.assertEqual(rendered.deviceIndependentSize().toSize(), size)
            widget.close()

    def test_pptx_round_trip_images_tables_and_assumptions(self):
        s = snapshot()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "report.pptx"
            save_pptx(output, s)
            report = Presentation(output)
            self.assertGreaterEqual(len(report.slides), 11)
            pictures = [shape for slide in report.slides for shape in slide.shapes
                        if shape.shape_type == 13]
            self.assertEqual(len(pictures), 6)
            for picture in pictures:
                self.assertAlmostEqual(picture.width / picture.height, 1760 / 992, places=4)
            tables = [shape for slide in report.slides for shape in slide.shapes if shape.has_table]
            self.assertEqual(len(tables), 4)
            text = "\n".join(shape.text for slide in report.slides for shape in slide.shapes
                             if shape.has_text_frame)
            self.assertIn("모델 가정", text)
            self.assertIn("파장의 1/10", text)
            self.assertIn("Re(Zc)", text)
            with zipfile.ZipFile(output) as package:
                self.assertIsNone(package.testzip())

    def test_atomic_exports_preserve_existing_files_when_writing_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "existing.png"
            output.write_bytes(b"existing content")
            with patch("zcalc.visuals._image", return_value=SimpleNamespace(save=lambda *args: False)):
                with self.assertRaises(OSError):
                    save_png(output, snapshot(), "impedance")
            self.assertEqual(output.read_bytes(), b"existing content")
            pptx_output = Path(folder) / "existing.pptx"
            pptx_output.write_bytes(b"existing report")
            with patch("pptx.presentation.Presentation.save", side_effect=OSError("simulated write error")):
                with self.assertRaises(OSError):
                    save_pptx(pptx_output, snapshot())
            self.assertEqual(pptx_output.read_bytes(), b"existing report")
            self.assertEqual({path.name for path in Path(folder).iterdir()}, {output.name, pptx_output.name})


if __name__ == "__main__":
    unittest.main()
