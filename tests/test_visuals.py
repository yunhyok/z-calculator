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
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from zcalc import engine, visuals
from zcalc.engine import Geometry
from zcalc.visuals import (LOG_X, METRICS, PlotWidget, Section, SectionWidget, Snapshot, _at, _at_design,
                           _FrequencyAxis, _image, _linear_ticks, _title, _values, _x_ticks, _y_ticks,
                           save_png, save_pptx)


def snapshot(kind="coupled_stripline", shield=True):
    f = np.geomspace(1e6, 1e10, 41)
    result = SimpleNamespace(
        impedance_ohm=100 + np.linspace(2, -2, f.size),
        s11_db=-20 - 3 * np.sin(np.linspace(0, 8, f.size)),
        attenuation_db_m=np.sqrt(f / 1e9),
        s21_db=-.1 * np.sqrt(f / 1e9),
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
        for conductivity in (0, -1, np.inf, np.nan):
            with self.subTest(conductivity=conductivity), self.assertRaises(ValueError):
                replace(s, conductivity_s_m=conductivity)
        with self.assertRaises(ValueError):
            replace(s, conductor_temperature_c=np.nan)

    def test_conductor_metadata_defaults_and_section_propagation(self):
        s = snapshot()
        self.assertEqual((s.conductor_name, s.conductivity_s_m, s.conductor_temperature_c),
                         ("구리", engine.COPPER_CONDUCTIVITY_S_M, 20))
        other = replace(s, conductor_name="합성 알루미늄", conductivity_s_m=3.5e7,
                        conductor_temperature_c=65)
        self.assertEqual((other.section.conductor_name, other.section.conductivity_s_m,
                          other.section.conductor_temperature_c), ("합성 알루미늄", 3.5e7, 65))
        pending = replace(other.section, conductivity_s_m=None)
        self.assertIn("입력 필요", visuals._conductor_label(pending))
        self.assertNotEqual(_image(other.section, "section"), _image(s.section, "section"))

    def test_non_copper_png_and_pptx_record_applied_material_without_stale_assumptions(self):
        s = replace(snapshot(), conductor_name="합성 알루미늄", conductivity_s_m=3.5e7,
                    conductor_temperature_c=65)
        with tempfile.TemporaryDirectory() as folder:
            for metric in ("dcr", "section"):
                path = Path(folder) / f"{metric}.png"
                with patch("zcalc.visuals._text", wraps=visuals._text) as draw_text:
                    save_png(path, s, metric)
                labels = "\n".join(str(call.args[5]) for call in draw_text.call_args_list)
                self.assertIn("합성 알루미늄", labels)
                self.assertIn("3.5e+07 S/m", labels)
                self.assertIn("기준 온도 65°C", labels)
                png = QImage(str(path))
                self.assertEqual(png.text("Conductor"), "합성 알루미늄")
                self.assertEqual(float(png.text("ConductivitySPerM")), 3.5e7)
                self.assertEqual(png.text("ConductorSourceTemperatureC"), "65")
            for temperature in (65, None):
                with self.subTest(temperature=temperature):
                    material = replace(s, conductor_temperature_c=temperature)
                    path = Path(folder) / "material.pptx"
                    save_pptx(path, material)
                    text = "\n".join(shape.text for slide in Presentation(path).slides
                                     for shape in slide.shapes if shape.has_text_frame)
                    table_text = "\n".join(cell.text for slide in Presentation(path).slides
                                           for shape in slide.shapes if shape.has_table
                                           for row in shape.table.rows for cell in row.cells)
                    self.assertIn("합성 알루미늄", text)
                    self.assertIn("3.5e+07 S/m", text)
                    self.assertIn("μr = 1", text)
                    self.assertIn("신호·접지", text)
                    self.assertIn("온도 보정 없이", text)
                    self.assertIn("기준 온도 " + ("65°C" if temperature is not None else "미지정"), text)
                    self.assertNotIn("구리", text + table_text)
                    self.assertNotIn("동박", text + table_text)
                    self.assertNotIn("20°C", text + table_text)

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
        self.assertEqual({key for key, log in LOG_X.items() if not log}, {"s11", "attenuation", "s21"})
        s = snapshot()
        for metric in METRICS:
            with self.subTest(metric=metric):
                default = _image(s, metric)
                self.assertEqual(default, _image(s, metric, LOG_X[metric]))
                self.assertNotEqual(default, _image(s, metric, not LOG_X[metric]))
        widget = PlotWidget()
        widget.set_metric("s11")
        widget.set_log_x(True)
        widget.set_metric("s21")
        self.assertEqual((widget.log_x["s11"], widget.log_x["s21"]), (True, False))

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

    def test_compact_widget_and_perfect_match(self):
        s = snapshot()
        s.result.s11_db = np.full(s.freq_hz.shape, -np.inf)
        for widget in (SectionWidget(), PlotWidget()):
            if isinstance(widget, PlotWidget):
                widget.set_metric("s11")
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

    def test_s_parameter_graphs_are_negative_db_with_a_zero_ceiling(self):
        s = snapshot()
        low, high, ticks = _y_ticks(_values(s, "s21"))
        self.assertEqual(high, 0)
        self.assertLess(low, 0)
        self.assertTrue(np.all(_values(s, "s11") < 0))
        self.assertEqual((_title(s, "s11"), _title(s, "s21")), ("차동 반사 Sdd11", "차동 전송 Sdd21"))
        single = snapshot("microstrip")
        self.assertEqual((_title(single, "s11"), _title(single, "s21")), ("반사 S11", "전송 S21"))

    def test_movable_marker_reads_the_curve_and_is_exported(self):
        s = snapshot()
        plain = _image(s, "s21")
        self.assertNotEqual(_image(s, "s21", marker_hz=3e9), plain)
        self.assertNotEqual(_image(s, "s21", marker_hz=3e9), _image(s, "s21", marker_hz=6e9))
        self.assertEqual(_image(s, "s21", marker_hz=1e12), plain)  # outside the sweep: not drawn
        self.assertAlmostEqual(_at(s, _values(s, "attenuation"), 4e9), 2, delta=0.02)
        widget = PlotWidget()
        widget.set_metric("s21")
        widget.set_snapshot(s)
        for size in (QSize(900, 300), QSize(1100, 620)):  # compact and full drawing
            widget.resize(size)
            widget.set_marker(None)
            widget.grab()
            rect, axis = widget._plot
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=rect.center().toPoint())
            self.assertAlmostEqual(widget.marker_hz, axis.frequency(.5), delta=(axis.high - axis.low) / rect.width())
            # Dragging past the edge stops at the end of the sweep; a right click removes the marker.
            QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=rect.center().toPoint())
            QTest.mouseMove(widget, QPoint(int(rect.right()) + 40, int(rect.center().y())))
            QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(int(rect.right()) + 40, int(rect.center().y())))
            self.assertEqual(widget.marker_hz, s.freq_hz[-1])
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))  # outside the plot area
            self.assertEqual(widget.marker_hz, s.freq_hz[-1])
            QTest.mouseClick(widget, Qt.MouseButton.RightButton, pos=rect.center().toPoint())
            self.assertIsNone(widget.marker_hz)
        widget.close()
        with tempfile.TemporaryDirectory() as folder:
            save_pptx(Path(folder) / "marker.pptx", s, marker_hz=2.5e9)
            text = "\n".join(cell.text for slide in Presentation(Path(folder) / "marker.pptx").slides
                             for shape in slide.shapes if shape.has_table
                             for row in shape.table.rows for cell in row.cells)
            self.assertIn("마커 2.5 GHz", text)
            self.assertIn("Sdd21 (dB)", text)

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
