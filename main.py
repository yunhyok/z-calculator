import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from zcalc import __version__
from zcalc.ui import Window, configure_style


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("ZCalculator")
    app.setApplicationName("Z Calculator")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "zcalc.ico")))
    configure_style(app)
    window = Window()
    if len(sys.argv) == 2:
        window.open_library(Path(sys.argv[1]))
    window.show()
    QTimer.singleShot(0, window.calculate)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
