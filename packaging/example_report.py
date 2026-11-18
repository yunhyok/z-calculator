"""Write the example report attached to a release, through the application itself.

Usage: python packaging/example_report.py OUTPUT.pptx

The report is the width synthesis of a 100 ohm differential stripline with the
application defaults: manual dielectric (Dk 4.2, Df 0.018), 1 GHz, 100 mm.
"""
import os
from pathlib import Path
import sys
import tempfile
import time

if sys.platform != "win32":
    # Windows draws with its own platform and fonts; Qt's offscreen platform has no fonts there.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from zcalc.ui import Window, configure_style
from zcalc.visuals import save_pptx


def main(output):
    app = QApplication([])
    if not QFontDatabase.families(QFontDatabase.WritingSystem.Korean):
        raise SystemExit("No font with Korean glyphs: the figures in the report would be unreadable.")
    configure_style(app)
    with tempfile.TemporaryDirectory() as folder:
        # Keep the user's stored library path and export folder out of the example.
        app.setOrganizationName("ZCalculatorExample")
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, folder)
        window = Window()
        window.kind.setCurrentIndex(window.kind.findData("coupled_stripline"))
        window.mode.setCurrentIndex(1)
        window.calculate()
        deadline = time.monotonic() + 120
        while window.job is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if window.job is not None or window.dirty or window.snapshot is None:
            raise SystemExit(f"Calculation failed: {window.status.text()}")
        save_pptx(output, window.snapshot, window.plot.log_x)
        width = window.snapshot.geometry.width_um
        window.close()
    # ASCII only: a build console may not encode the Korean status text.
    print(f"{output}: synthesized bottom width {width:.3f} um")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
