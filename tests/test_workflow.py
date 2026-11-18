import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog
from pptx import Presentation

from zcalc.ui import MANUAL, Window, configure_style

CMX = ('<Cadence_Material_Lib><Material name="Demo"><Dielectric><Model>'
       '100 4.3 0.015\n10000 4 0.02</Model></Dielectric></Material></Cadence_Material_Lib>')


class DesktopWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_style(cls.app)

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        settings = QSettings(str(self.folder / "settings.ini"), QSettings.Format.IniFormat)
        with patch("zcalc.ui.QSettings", return_value=settings):
            self.window = Window()
        self.addCleanup(self.window.close)
        self.cmx = self.folder / "sample.cmx"
        self.cmx.write_text(CMX)

    def calculate(self):
        self.window.calculate()
        deadline = time.monotonic() + 30
        while self.window.job is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(self.window.job)
        self.assertTrue(self.window.report_button.isEnabled(), self.window.status.text())

    def save(self, name, *arguments, **keywords):
        path = self.folder / name
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")):
            self.window.export(*arguments, **keywords)
        return path

    def select(self, combo, data):
        combo.setCurrentIndex(combo.findData(data))

    def test_import_synthesize_plot_export(self):
        self.window.open_library(self.cmx)
        self.select(self.window.kind, "coupled_stripline")
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
        self.assertEqual(self.window.section.section.geometry, snap.geometry)
        for i in range(self.window.metric_selector.count()):
            self.window.metric_selector.setCurrentIndex(i)
            path = self.save(self.window.metric_selector.currentData() + ".png", False)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        slides = Presentation(self.save("report.pptx", True)).slides
        self.assertGreaterEqual(len(slides), 6)
        self.assertTrue(any("목표 100 Ω" in shape.text for shape in slides[0].shapes if shape.has_text_frame))
        self.window.fields["start"].setText("50")
        self.assertFalse(self.window.report_button.isEnabled())
        self.window.calculate()
        self.assertIsNone(self.window.job)
        self.assertIn("주파수", self.window.status.text())

    def test_section_follows_inputs_before_calculation(self):
        window, fields = self.window, self.window.fields
        drawn = lambda: window.section.section.geometry
        self.assertIsNone(window.snapshot)
        self.assertEqual(drawn().kind, "microstrip")
        self.select(window.kind, "coupled_stripline")
        self.assertEqual((drawn().kind, drawn().gap_um, drawn().top_um), ("coupled_stripline", 120, 100))
        fields["gap_um"].setText("75")
        window.shield.setChecked(True)
        fields["clearance_um"].setText("60")
        fields["angle_deg"].setText("70")
        self.assertEqual((drawn().gap_um, drawn().shield, drawn().clearance_um, drawn().angle_deg),
                         (75, True, 60, 70))
        self.assertIsNone(window.snapshot)
        # Half-typed text keeps the last usable drawing and marks the field.
        fields["gap_um"].setText("")
        self.assertEqual(drawn().gap_um, 75)
        self.assertTrue(fields["gap_um"].property("invalid"))
        self.assertFalse(window.section_save.isEnabled())
        fields["gap_um"].setText("80")
        self.assertFalse(fields["gap_um"].property("invalid"))
        self.assertEqual(drawn().gap_um, 80)
        path = self.save("section.png", False, section=True)
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        # A hidden field does not block the drawing; a width to be synthesized is not dimensioned.
        self.select(window.kind, "microstrip")
        fields["gap_um"].setText("oops")
        self.assertFalse(fields["gap_um"].property("invalid"))
        self.assertTrue(window.section_save.isEnabled())
        window.mode.setCurrentIndex(1)
        self.assertFalse(window.section.section.width_known)
        self.assertFalse(window.section_save.isEnabled())

    def test_library_values_stay_editable(self):
        window, editor = self.window, self.window.bottom
        window.show()
        window.open_library(self.cmx)
        # Loading a library keeps the manual values; picking an entry shows it at the design frequency.
        self.assertEqual((editor.selector.currentData(), editor.er.text()), (None, "4.2"))
        editor.selector.setCurrentIndex(1)
        self.assertEqual((editor.er.text(), editor.df.text()), ("4.15", "0.0175"))
        self.assertFalse(editor.er.isHidden() or not editor.er.isEnabled())
        section = window.section.section
        self.assertEqual((section.bottom_name, section.bottom_er), ("Demo", 4.15))
        window.fields["design"].setText("10")
        self.assertEqual((editor.selector.currentData(), editor.er.text(), editor.df.text()), ("Demo", "4", "0.02"))
        self.calculate()
        self.assertEqual(window.snapshot.bottom_name, "Demo")
        self.assertGreater(np.ptp(window.snapshot.bottom_er), 0.25)
        # Typing over a library value switches to manual entry with the other value kept.
        QTest.keyClicks(editor.er, ".1")
        self.assertEqual((editor.selector.currentData(), editor.er.text(), editor.df.text()), (None, "4.1", "0.02"))
        self.assertEqual(window.section.section.bottom_name, MANUAL)
        self.assertFalse(window.report_button.isEnabled())
        self.calculate()
        snap = window.snapshot
        self.assertEqual(snap.bottom_name, MANUAL)
        self.assertTrue(np.all(snap.bottom_er == 4.1) and np.all(snap.bottom_df == 0.02))
        # Returning to the library entry restores its table values.
        editor.selector.setCurrentIndex(1)
        self.assertEqual((editor.er.text(), editor.df.text()), ("4", "0.02"))
        self.assertEqual(window.section.section.bottom_name, "Demo")

    def test_unusable_input_is_named_marked_and_focused(self):
        window = self.window
        window.show()
        window.fields["length"].setText("-5")
        self.assertTrue(window.fields["length"].property("invalid"))
        window.calculate()
        self.assertIsNone(window.job)
        self.assertIn("선로 길이", window.status.text())
        self.assertEqual(window.tabs.currentIndex(), 2)
        window.fields["length"].setText("100")
        self.assertFalse(window.fields["length"].property("invalid"))
        window.bottom.er.setText("0.5")
        self.assertTrue(window.bottom.er.property("invalid"))
        window.calculate()
        self.assertIsNone(window.job)
        self.assertIn("Dk", window.status.text())
        self.assertEqual(window.tabs.currentIndex(), 1)
        window.bottom.er.setText("4.2")
        window.fields["design"].setText("30")
        window.calculate()
        self.assertIn("설계 주파수", window.status.text())
        self.assertEqual(window.tabs.currentIndex(), 2)
        # A successful calculation returns the keyboard to the field being edited.
        window.fields["design"].setText("1")
        self.assertIs(window.focusWidget(), window.fields["design"])
        self.calculate()
        self.assertIs(window.focusWidget(), window.fields["design"])

    def test_frequency_axis_per_graph_and_stale_results(self):
        window = self.window
        self.calculate()
        self.assertFalse(window.plot.stale)
        self.assertTrue(window.stale_note.isHidden())
        # Loss graphs open on a linear frequency axis; each graph remembers its own choice.
        expected = {"impedance": True, "dcr": True, "return": False, "attenuation": False, "insertion": False}
        for key, log in expected.items():
            self.select(window.metric_selector, key)
            self.assertEqual(window.log_axis.isChecked(), log, key)
        window.log_axis.setChecked(True)
        self.assertTrue(window.plot.log_x["insertion"])
        self.select(window.metric_selector, "return")
        self.assertFalse(window.log_axis.isChecked())
        self.select(window.metric_selector, "insertion")
        self.assertTrue(window.log_axis.isChecked())
        linear = np.diff(window.snapshot.freq_hz)
        self.assertLess(linear.max(), (window.snapshot.freq_hz[-1] - window.snapshot.freq_hz[0]) / 399)
        # Results of earlier inputs stay visible but are marked, and cannot be exported.
        window.fields["width_um"].setText("200")
        self.assertTrue(window.plot.stale)
        self.assertFalse(window.stale_note.isHidden())
        self.assertFalse(window.png_button.isEnabled() or window.report_button.isEnabled())
        self.assertTrue(window.section_save.isEnabled())
        self.assertEqual(window.section.section.geometry.width_um, 200)
        self.calculate()
        self.assertFalse(window.plot.stale)
        self.assertTrue(window.stale_note.isHidden())
        self.assertTrue(window.png_button.isEnabled())
