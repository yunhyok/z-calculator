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

from zcalc import engine, visuals
from zcalc.engine import Geometry
from zcalc.visuals import (LOG_X, METRICS, PlotWidget, Section, SectionWidget, Snapshot, _at_design,
                           _FrequencyAxis, _image, _linear_ticks, _x_ticks, _y_ticks, save_png, save_pptx)


def snapshot(kind="coupled_stripline", shield=True):
    f = np.geomspace(1e6, 1e10, 41)
    result = SimpleNamespace(
        impedance_ohm=100 + np.linspace(2, -2, f.size),
        return_loss_db=20 + 3 * np.sin(np.linspace(0, 8, f.size)),
        attenuation_db_m=np.sqrt(f / 1e9),
        insertion_loss_db=.1 * np.sqrt(f / 1e9),
        dcr_ohm_m=2.1,
        warnings=("wavelength",),
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

    def test_linear_frequency_axis_uses_round_ticks(self):
        # A 10 MHz start is within 5% of the span from zero, so the axis starts at zero.
        self.assertEqual(_linear_ticks(1e7, 2e10)[:3], (0, 2e10, [0, 5e9, 1e10, 1.5e10, 2e10]))
        self.assertEqual(_linear_ticks(1e7, 2e10)[4], 5e9)
        axis = _FrequencyAxis(np.array([1e7, 2e10]), log=False)
        self.assertEqual((axis.labels, axis.title), (["0", "5", "10", "15", "20"], "주파수 (GHz)"))
        self.assertAlmostEqual(axis.position(1e10), 0.5)
        # A start far from a round bound keeps the data range: no empty band left of the curve.
        start, end, major, _, _ = _linear_ticks(3.3e8, 9.1e8)
        self.assertEqual((start, end), (3.3e8, 9.1e8))
        self.assertTrue(all(start <= tick <= end for tick in major))
        axis = _FrequencyAxis(np.array([3.3e8, 9.1e8]), log=False)
        self.assertEqual((axis.labels, axis.title), (["400", "600", "800"], "주파수 (MHz)"))
        # Narrow sweeps and 2.5 steps still get distinct labels.
        narrow = _FrequencyAxis(np.array([1e9, 1.00001e9]), log=False)
        self.assertEqual(len(set(narrow.labels)), len(narrow.labels))
        self.assertEqual(narrow.labels[0], "1")
        self.assertEqual(_FrequencyAxis(np.array([1e9, 1.2e10]), log=False).labels, ["2.5", "5", "7.5", "10", "12.5"])
        # A sweep one rounding step wide draws instead of dividing by zero.
        point = _FrequencyAxis(np.array([1e9, np.nextafter(1e9, 2e9)]), log=False)
        self.assertTrue(0 <= point.position(1e9) <= 1)
        point.high = point.low
        self.assertEqual(point.position(1e9), .5)
        log_axis = _FrequencyAxis(np.array([1e6, 1e10]), log=True)
        self.assertAlmostEqual(log_axis.position(1e8), 0.5)
        self.assertEqual(log_axis.labels[0], "1 MHz")

    def test_loss_graphs_default_to_linear_frequency_and_can_switch(self):
        self.assertEqual({key for key, log in LOG_X.items() if not log}, {"return", "attenuation", "insertion"})
        s = snapshot()
        for metric in METRICS:
            with self.subTest(metric=metric):
                default = _image(s, metric)
                self.assertEqual(default, _image(s, metric, LOG_X[metric]))
                self.assertNotEqual(default, _image(s, metric, not LOG_X[metric]))
        widget = PlotWidget()
        widget.set_metric("return")
        widget.set_log_x(True)
        widget.set_metric("insertion")
        self.assertEqual((widget.log_x["return"], widget.log_x["insertion"]), (True, False))

    def test_section_is_drawn_from_inputs_without_a_result(self):
        section = Section(Geometry(kind="coupled_microstrip", shield=True), "직접 입력", "공기", 4.2, 1.0)
        pending = replace(section, width_known=False, bottom_er=None)
        self.assertNotEqual(_image(section, "section"), _image(pending, "section"))
        self.assertEqual(_image(snapshot(), "section"), _image(snapshot().section, "section"))
        widget = SectionWidget()
        widget.resize(550, 210)
        blank = widget.grab().toImage()
        widget.set_section(pending)
        self.assertNotEqual(widget.grab().toImage(), blank)
        widget.close()

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
            else:
                widget.set_section(s.section)
            for size in (QSize(550, 210), QSize(820, 245), QSize(1000, 564)):
                widget.resize(size)
                rendered = widget.grab().toImage()
                self.assertFalse(rendered.isNull())
                self.assertEqual(rendered.deviceIndependentSize().toSize(), size)
            widget.close()

    def test_every_engine_warning_has_display_text(self):
        self.assertEqual(set(visuals.WARNINGS), set(engine.WARNINGS))
        self.assertEqual(set(visuals.KINDS), set(engine.KINDS))

    def test_stale_plot_is_marked(self):
        widget = PlotWidget()
        widget.set_snapshot(snapshot())
        for size in (QSize(820, 245), QSize(1000, 564)):
            widget.resize(size)
            widget.set_stale(False)
            fresh = widget.grab().toImage()
            widget.set_stale(True)
            self.assertNotEqual(widget.grab().toImage(), fresh)
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
