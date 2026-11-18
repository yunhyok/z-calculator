"""Windows desktop workspace for stackup analysis."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .engine import C0, Geometry, analyze, synthesize_width
from .materials import load_cmx, manual_material
from .visuals import PlotWidget, SectionWidget, Snapshot, save_png, save_pptx


KINDS = {
    "microstrip": "Microstrip", "stripline": "Stripline",
    "coupled_microstrip": "Coupled microstrip", "coupled_stripline": "Coupled stripline",
}
METRICS = {
    "impedance": "특성 임피던스", "dcr": "단위 길이당 DCR", "return": "반사 손실",
    "attenuation": "단위 길이당 전파 손실", "insertion": "지정 길이 삽입 손실",
}


def configure_style(app):
    app.setStyle("Fusion")
    palette = app.palette()
    for role, color in {
        "Window": "#202226", "WindowText": "#eceef1", "Base": "#1d2025",
        "AlternateBase": "#2b2e34", "Text": "#eceef1", "Button": "#353942",
        "ButtonText": "#eceef1", "Highlight": "#386cb0", "HighlightedText": "#ffffff",
        "ToolTipBase": "#303741", "ToolTipText": "#eceef1", "PlaceholderText": "#aab2bd",
    }.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(color))
    app.setPalette(palette)
    font = app.font()
    font.setFamily("Malgun Gothic")
    font.setPointSizeF(9.5)
    app.setFont(font)
    app.setStyleSheet("""
        QMainWindow, QWidget#root { background: #202226; }
        QFrame#header, QFrame#panel, QFrame#stat { background: #2b2e34; }
        QFrame#panel, QFrame#stat { border: 1px solid #424750; border-radius: 7px; }
        QLabel { background: transparent; }
        QLabel#brand { font-family: 'Segoe UI'; font-size: 23px; font-weight: 650; }
        QLabel#subtle, QLabel#unit { color: #b8c0cb; }
        QLabel#sectionTitle { font-size: 14px; font-weight: bold; color: #e9edf2; }
        QLabel#value { color: #f1f5fa; font-family: 'Segoe UI'; font-size: 22px; font-weight: 600; }
        QLabel#primaryValue { color: #9ecaff; font-family: 'Segoe UI'; font-size: 22px; font-weight: 600; }
        QLabel#status { color: #b9c7d9; }
        QLabel#status[error="true"] { color: #ffb6ab; }
        QLineEdit, QComboBox { background: #1d2025; border: 1px solid #525b67;
            border-radius: 4px; padding: 3px 7px; min-height: 21px; }
        QLineEdit:focus, QComboBox:focus, QPushButton:focus, QCheckBox:focus {
            border: 1px solid #8fc1ff; }
        QLineEdit:disabled { color: #a3acb7; background: #292d32; }
        QComboBox QAbstractItemView { background: #23272c; selection-background-color: #386cb0; }
        QPushButton { background: #393e46; color: #ecf0f4; border: 1px solid #575f6a;
            border-radius: 5px; padding: 5px 12px; }
        QPushButton:hover { background: #454d58; }
        QPushButton#primary { background: #386cb0; color: white; border-color: #5588c7;
            font-weight: bold; min-height: 24px; }
        QPushButton#primary:hover { background: #427bc3; }
        QPushButton:disabled { color: #99a3ae; background: #2c3036; border-color: #444951; }
        QTabWidget::pane { border: none; border-top: 1px solid #50555f; }
        QTabBar::tab { background: #292d33; color: #c5cbd2; padding: 8px 14px; }
        QTabBar::tab:selected { color: #eaf2ff; border-bottom: 2px solid #8fc1ff; }
        QScrollArea { background: transparent; border: none; }
        QSplitter::handle { background: #202226; }
        QScrollBar:vertical { background: #25282e; width: 12px; }
        QScrollBar::handle:vertical { background: #555e6a; min-height: 30px; border-radius: 5px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    """)


def label(text, style=None):
    widget = QLabel(text)
    if style:
        widget.setObjectName(style)
    return widget


def field(value, name):
    edit = QLineEdit(str(value))
    edit.setAccessibleName(name)
    edit.setAlignment(Qt.AlignmentFlag.AlignRight)
    edit.setMinimumWidth(54)
    return edit


def number(edit):
    try:
        value = float(edit.text().strip())
        if not np.isfinite(value):
            raise ValueError
        return value
    except ValueError as exc:
        raise ValueError(f"{edit.accessibleName()}에 유한한 숫자를 입력하세요.") from exc


def form():
    layout = QFormLayout()
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setVerticalSpacing(6)
    layout.setHorizontalSpacing(12)
    return layout


class MaterialEditor(QWidget):
    changed = Signal()

    def __init__(self, title):
        super().__init__()
        self.library = {}
        self.title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(label(title, "sectionTitle"))
        self.selector = QComboBox()
        self.selector.setAccessibleName(title)
        self.selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.selector.addItem("직접 입력", None)
        layout.addWidget(self.selector)
        row = QHBoxLayout()
        self.er, self.df = field("4.2", title + " Dk"), field("0.018", title + " Df")
        for name, edit in [("Dk", self.er), ("Df", self.df)]:
            row.addWidget(label(name, "subtle"))
            row.addWidget(edit, 1)
            edit.textChanged.connect(self.changed)
        layout.addLayout(row)
        self.details = label("", "subtle")
        self.details.setWordWrap(True)
        self.details.hide()
        layout.addWidget(self.details)
        self.selector.currentIndexChanged.connect(self._selection)

    def set_library(self, library):
        previous = self.selector.currentData()
        self.library = library
        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItem("직접 입력", None)
        for name in sorted(library, key=str.casefold):
            self.selector.addItem(name, name)
        self.selector.setCurrentIndex(max(0, self.selector.findData(previous)) if previous else 0)
        self.selector.blockSignals(False)
        self._selection()

    def _selection(self):
        is_manual = self.selector.currentIndex() == 0
        self.er.setVisible(is_manual)
        self.df.setVisible(is_manual)
        for widget in self.findChildren(QLabel):
            if widget.text() in ("Dk", "Df"):
                widget.setVisible(is_manual)
        self.details.setVisible(not is_manual)
        self.changed.emit()

    def sample(self, frequencies):
        name = self.selector.currentData()
        material = self.library[name] if name else manual_material(number(self.er), number(self.df))
        er, df = material.sample(frequencies)
        return name or "직접 입력", er, df

    def update_detail(self, design_hz):
        name = self.selector.currentData()
        if name and design_hz > 0:
            material = self.library[name]
            er, df = material.sample(np.array([design_hz]))
            lo, hi = material.frequency_range_hz
            reference = (f"{lo / 1e6:g} MHz 단일 기준점 · 상수 적용" if material.is_constant else
                         f"{lo / 1e6:g}–{hi / 1e6:g} MHz · 범위 밖 끝값 고정")
            temp = "기본 온도 미지정" if material.temperature_c is None else f"{material.temperature_c:g} °C"
            self.details.setText(f"{design_hz / 1e9:g} GHz   ·   Dk {er[0]:.5g}   /   Df {df[0]:.4g}"
                                 f"\n{reference}\n{temp}")


class Calculation(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, inputs, target, parent):
        super().__init__(parent)
        self.inputs, self.target = inputs, target

    def run(self):
        try:
            p = self.inputs
            geom = p["geometry"]
            index = int(np.argmin(abs(p["freq_hz"] - p["design_hz"])))
            props = [p[key] for key in ("bottom_er", "bottom_df", "top_er", "top_df")]
            if self.target is not None:
                width = synthesize_width(geom, self.target, p["design_hz"], *(v[index] for v in props))
                geom = replace(geom, width_um=width)
            result = analyze(geom, p["freq_hz"], *props, p["length_m"], p["reference_ohm"])
            self.ready.emit(Snapshot(**{**p, "geometry": geom}, result=result, target_ohm=self.target))
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z Calculator")
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1260, available.width() - 30), min(840, available.height() - 64))
        self.settings = QSettings()
        self.snapshot = None
        self.job = None
        self.dirty = True
        self.pair_mode = False
        self.fields = {}
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        body = QVBoxLayout(root)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("header")
        head = QHBoxLayout(bar)
        head.setContentsMargins(16, 9, 16, 9)
        icon = QLabel()
        icon.setPixmap(QApplication.windowIcon().pixmap(QSize(36, 36)))
        head.addWidget(icon)
        head.addWidget(label("Z Calculator", "brand"))
        self.subtitle = label("전송선 단면 해석", "subtle")
        head.addWidget(self.subtitle)
        head.addStretch()
        self.png_button = QPushButton("PNG 저장")
        self.report_button = QPushButton("PowerPoint 보고서")
        head.addWidget(self.png_button)
        head.addWidget(self.report_button)
        self.png_button.clicked.connect(lambda: self.export(False))
        self.report_button.clicked.connect(lambda: self.export(True))
        body.addWidget(bar)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        outer = QHBoxLayout()
        outer.setContentsMargins(12, 12, 12, 6)
        outer.addWidget(self.splitter)
        body.addLayout(outer, 1)
        self.sidebar = self.make_sidebar()
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.make_results())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([330, 886])
        self.status = label("해석 준비", "status")
        self.status.setWordWrap(True)
        self.status.setContentsMargins(16, 2, 16, 7)
        body.addWidget(self.status)
        self.kind.currentIndexChanged.connect(self.kind_changed)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.shield.toggled.connect(self.geometry_visibility)
        for edit in self.fields.values():
            edit.textChanged.connect(self.mark_dirty)
            edit.returnPressed.connect(self.calculate)
        for editor in (self.bottom, self.top):
            editor.changed.connect(self.mark_dirty)
        self.geometry_visibility()
        self.mode_changed()
        self.shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut.activated.connect(self.calculate)
        recent = self.settings.value("materials/lastPath", "", type=str)
        if recent and Path(recent).is_file():
            self.open_library(Path(recent), quiet=True)
        self.mark_dirty()

    def add_field(self, layout, key, title, value):
        edit = field(value, title)
        self.fields[key] = edit
        layout.addRow(title, edit)
        return edit

    def scroll_page(self):
        area = QScrollArea()
        area.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(9)
        area.setWidget(page)
        return area, layout

    def make_sidebar(self):
        pane = QFrame()
        pane.setObjectName("panel")
        pane.setMinimumWidth(260)
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(3, 3, 3, 8)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        stack, stack_layout = self.scroll_page()
        self.tabs.addTab(stack, "단면")
        material, mat_layout = self.scroll_page()
        self.tabs.addTab(material, "재료")
        sweep, sweep_layout = self.scroll_page()
        self.tabs.addTab(sweep, "해석 조건")
        stack_layout.addWidget(label("전송선 구조", "sectionTitle"))
        self.kind = QComboBox()
        self.kind.setAccessibleName("전송선 구조")
        for key, name in KINDS.items():
            self.kind.addItem(name, key)
        stack_layout.addWidget(self.kind)
        self.mode = QComboBox()
        self.mode.setAccessibleName("계산 방식")
        self.mode.addItems(["치수 → 임피던스", "목표 임피던스 → 선폭"])
        stack_layout.addWidget(self.mode)
        stack_layout.addSpacing(4)
        stack_layout.addWidget(label("도체 치수", "sectionTitle"))
        self.geo_form = form()
        stack_layout.addLayout(self.geo_form)
        for key, title, value in [
            ("target", "목표 임피던스 (Ω)", 50), ("width_um", "하부 선폭 W (µm)", 180),
            ("thickness_um", "구리 두께 T (µm)", 18), ("angle_deg", "측벽 각도 (°)", 90),
            ("gap_um", "도선 간격 S (µm)", 120),
        ]:
            self.add_field(self.geo_form, key, title, value)
        stack_layout.addSpacing(4)
        stack_layout.addWidget(label("기준면과 차폐", "sectionTitle"))
        self.layer_form = form()
        stack_layout.addLayout(self.layer_form)
        self.add_field(self.layer_form, "bottom_um", "하부 유전체 H₁ (µm)", 100)
        self.add_field(self.layer_form, "top_um", "상부 유전체 H₂ (µm)", 100)
        self.shield = QCheckBox("동일층 양측 접지 실드")
        self.fields["angle_deg"].setToolTip("기판 면 기준 45–90°. 90°는 직사각형 도선입니다.")
        stack_layout.addWidget(self.shield)
        self.shield_form = form()
        stack_layout.addLayout(self.shield_form)
        self.add_field(self.shield_form, "clearance_um", "실드 간격 G (µm)", 150)
        self.geometry_note = label("", "subtle")
        self.geometry_note.setWordWrap(True)
        stack_layout.addWidget(self.geometry_note)
        stack_layout.addStretch()

        mat_layout.addWidget(label("유전체 라이브러리", "sectionTitle"))
        self.library_button = QPushButton("CMX 파일 열기…")
        self.library_button.clicked.connect(self.browse_library)
        mat_layout.addWidget(self.library_button)
        self.library_info = label("로컬 CMX를 열거나 Dk·Df를 직접 입력하세요.", "subtle")
        self.library_info.setWordWrap(True)
        mat_layout.addWidget(self.library_info)
        self.bottom, self.top = MaterialEditor("하부 유전체"), MaterialEditor("상부 유전체")
        mat_layout.addSpacing(5)
        mat_layout.addWidget(self.bottom)
        mat_layout.addSpacing(5)
        mat_layout.addWidget(self.top)
        mat_layout.addStretch()

        sweep_layout.addWidget(label("주파수 범위", "sectionTitle"))
        grid = form()
        sweep_layout.addLayout(grid)
        for key, title, value in [("start", "시작 (GHz)", 0.01), ("stop", "종료 (GHz)", 20),
                                  ("design", "설계 주파수 (GHz)", 1)]:
            self.add_field(grid, key, title, value)
        sweep_layout.addSpacing(6)
        sweep_layout.addWidget(label("포트와 선로", "sectionTitle"))
        grid = form()
        sweep_layout.addLayout(grid)
        self.add_field(grid, "reference", "양단 기준 임피던스 (Ω)", 50)
        self.add_field(grid, "length", "선로 길이 (mm)", 100)
        self.port_note = label("", "subtle")
        self.port_note.setWordWrap(True)
        sweep_layout.addWidget(self.port_note)
        sweep_layout.addStretch()
        self.run_button = QPushButton("임피던스 계산")
        self.run_button.setObjectName("primary")
        self.run_button.setToolTip("Ctrl+Enter")
        self.run_button.clicked.connect(self.calculate)
        layout.addWidget(self.run_button)
        return pane

    def make_results(self):
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        heading = QHBoxLayout()
        heading.addWidget(label("설계점 결과", "sectionTitle"))
        heading.addStretch()
        self.condition = label("", "subtle")
        heading.addWidget(self.condition)
        layout.addLayout(heading)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.stat_labels, self.stat_values = [], []
        for i, caption in enumerate(["특성 임피던스", "도선 DCR", "반사 손실", "삽입 손실"]):
            box = QFrame()
            box.setObjectName("stat")
            col = QVBoxLayout(box)
            col.setContentsMargins(10, 7, 8, 8)
            col.setSpacing(2)
            title, value = label(caption, "subtle"), label("—", "primaryValue" if i == 0 else "value")
            value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            col.addWidget(title)
            col.addWidget(value)
            self.stat_labels.append(title)
            self.stat_values.append(value)
            row.addWidget(box, 1)
        layout.addLayout(row)
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(7)
        layout.addWidget(split, 1)
        self.section, self.plot = SectionWidget(), PlotWidget()
        for widget, title in [(self.section, "단면도"), (self.plot, "주파수 응답")]:
            pane = QFrame()
            pane.setObjectName("panel")
            col = QVBoxLayout(pane)
            col.setContentsMargins(8, 6, 8, 7)
            col.setSpacing(4)
            header = QHBoxLayout()
            header.addWidget(label(title, "sectionTitle"))
            header.addStretch()
            if widget is self.section:
                self.section_save = QPushButton("단면 PNG")
                self.section_save.clicked.connect(lambda: self.export(False, section=True))
                header.addWidget(self.section_save)
            else:
                self.metric_selector = QComboBox()
                self.metric_selector.setAccessibleName("그래프 항목")
                for key, name in METRICS.items():
                    self.metric_selector.addItem(name, key)
                self.metric_selector.currentIndexChanged.connect(lambda: self.plot.set_metric(self.metric_selector.currentData()))
                header.addWidget(self.metric_selector)
            col.addLayout(header)
            col.addWidget(widget, 1)
            split.addWidget(pane)
        split.setSizes([255, 350])
        return right

    def kind_changed(self):
        # Preserve a custom port or target when switching between SE and differential.
        pair = "coupled" in self.kind.currentData()
        if pair != self.pair_mode:
            for key in ("reference", "target"):
                if self.fields[key].text() == ("100" if self.pair_mode else "50"):
                    self.fields[key].setText("100" if pair else "50")
        self.pair_mode = pair
        self.geometry_visibility()

    def geometry_visibility(self):
        kind = self.kind.currentData()
        pair, strip = "coupled" in kind, "stripline" in kind
        self.geo_form.setRowVisible(self.fields["gap_um"], pair)
        self.layer_form.setRowVisible(self.fields["top_um"], strip)
        self.shield_form.setRowVisible(self.fields["clearance_um"], self.shield.isChecked())
        self.top.setVisible(strip)
        self.geometry_note.setText("상·하 기준면 거리는 도선 표면에서 측정합니다." if strip else
                                   "도선 위는 공기입니다. H₁은 하부 접지면까지의 거리입니다.")
        self.port_note.setText(("차동 포트" if pair else "단일 종단 포트") +
                               " 기준입니다. 반사·삽입 손실은 지정한 길이와 양단 임피던스를 사용합니다.")
        self.mark_dirty()

    def mode_changed(self):
        inverse = self.mode.currentIndex() == 1
        self.geo_form.setRowVisible(self.fields["target"], inverse)
        self.fields["width_um"].setEnabled(not inverse)
        self.run_button.setText("목표 선폭 계산" if inverse else "임피던스 계산")
        self.mark_dirty()

    def set_status(self, message, error=False):
        self.status.setText(message)
        self.status.setProperty("error", error)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def set_exports(self, enabled):
        for btn in (self.png_button, self.section_save, self.report_button):
            btn.setEnabled(enabled)

    def mark_dirty(self):
        self.dirty = True
        self.set_exports(False)
        if self.snapshot:
            self.set_status("입력이 변경되었습니다. 현재 표시된 결과를 갱신하려면 계산을 실행하세요.")
        try:
            hz = number(self.fields["design"]) * 1e9
            for editor in (self.bottom, self.top):
                editor.update_detail(hz)
        except ValueError:
            pass

    def browse_library(self):
        previous = self.settings.value("materials/lastPath", "", type=str)
        filename, _ = QFileDialog.getOpenFileName(self, "유전체 라이브러리 열기", previous,
                                                   "Cadence 재료 라이브러리 (*.cmx);;모든 파일 (*)")
        if filename:
            self.open_library(Path(filename))

    def open_library(self, path, quiet=False):
        try:
            library = load_cmx(path)
            if not library:
                raise ValueError("사용 가능한 유전체 항목이 없습니다.")
        except (ValueError, OSError) as exc:
            self.library_info.setText(f"파일 열기 실패 · {exc}")
            if not quiet:
                QMessageBox.warning(self, "CMX 파일 열기", str(exc))
            return
        for editor in (self.bottom, self.top):
            editor.set_library(library)
        self.library_info.setText(f"{path.name}   ·   유전체 {len(library)}개")
        self.library_info.setToolTip(str(path))
        self.settings.setValue("materials/lastPath", str(path.resolve()))

    def read_inputs(self):
        def value(key):
            return number(self.fields[key])
        kind = self.kind.currentData()
        pair, strip = "coupled" in kind, "stripline" in kind
        inverse = self.mode.currentIndex() == 1
        geometry = Geometry(kind=kind, width_um=value("width_um") if not inverse else 180.0,
                            thickness_um=value("thickness_um"), angle_deg=value("angle_deg"),
                            bottom_um=value("bottom_um"), top_um=value("top_um") if strip else 100.0,
                            gap_um=value("gap_um") if pair else 120.0, shield=self.shield.isChecked(),
                            clearance_um=value("clearance_um") if self.shield.isChecked() else 150.0)
        start, stop, design = (value(key) * 1e9 for key in ("start", "stop", "design"))
        if not 0 < start < stop or not start <= design <= stop:
            raise ValueError("시작과 종료 주파수를 오름차순으로 입력하고, 그 범위 안에 설계 주파수를 지정하세요.")
        length_m = value("length") / 1000
        if length_m < 0:
            raise ValueError("선로 길이는 0 mm 이상이어야 합니다.")
        if not 1e-3 <= start < stop <= 1e15:
            raise ValueError("해석 가능한 주파수 범위는 0.001 Hz–1 PHz입니다.")
        # Include material breakpoints and resolve the finite-line reflection ripple.
        points = [np.geomspace(start, stop, 321), [design]]
        max_er = 1.
        for editor in (self.bottom, self.top) if strip else (self.bottom,):
            name = editor.selector.currentData()
            if name:
                material = editor.library[name]
                knots = material.frequencies_hz
                active = np.r_[start, knots[(knots > start) & (knots < stop)], stop]
                points.append(active)
                max_er = max(max_er, float(material.sample(active)[0].max()))
            else:
                max_er = max(max_er, number(editor.er))
        intervals = (stop - start) / C0 * (32 * length_m) * np.sqrt(max_er)
        if intervals > 32000:
            raise ValueError("지정한 길이에서 반사 응답을 분해하려면 주파수 범위를 나누어 계산하세요 "
                             "(구간당 최대 32,000분할).")
        if intervals > 0:
            points.append(np.linspace(start, stop, int(np.ceil(intervals)) + 1))
        freq = np.unique(np.concatenate(points))
        bn, be, bd = self.bottom.sample(freq)
        if strip:
            tn, te, td = self.top.sample(freq)
        else:
            tn, te, td = "공기", np.ones_like(freq), np.zeros_like(freq)
        return dict(geometry=geometry, freq_hz=freq, bottom_name=bn, bottom_er=be, bottom_df=bd,
                    top_name=tn, top_er=te, top_df=td, length_m=length_m,
                    reference_ohm=value("reference"), design_hz=design), (value("target") if inverse else None)

    def calculate(self):
        if self.job is not None:
            return
        try:
            inputs, target = self.read_inputs()
        except ValueError as exc:
            self.set_status(str(exc), True)
            return
        self.job = Calculation(inputs, target, self)
        self.job.ready.connect(self.display_result)
        self.job.failed.connect(lambda message: self.set_status(message, True))
        self.job.finished.connect(self.calculation_finished)
        self.sidebar.setEnabled(False)
        self.set_exports(False)
        self.set_status("설계 주파수에서 목표 선폭을 찾는 중…" if target is not None else "단면 전계와 주파수 응답 계산 중…")
        self.job.start()

    def display_result(self, snap):
        self.snapshot = snap
        if self.mode.currentIndex() == 1:
            self.fields["width_um"].setText(f"{snap.geometry.width_um:.8g}")
        self.dirty = False
        self.section.set_snapshot(snap)
        self.plot.set_snapshot(snap)
        i = int(np.argmin(abs(snap.freq_hz - snap.design_hz)))
        r = snap.result
        pair = "coupled" in snap.geometry.kind
        self.stat_labels[0].setText("차동 임피던스" if pair else "특성 임피던스")
        self.stat_labels[1].setText("도선 DCR (1가닥)")
        for widget, v, unit in zip(self.stat_values, [r.impedance_ohm[i], r.dcr_ohm_m,
                                                     r.return_loss_db[i], r.insertion_loss_db[i]],
                                   ["Ω", "Ω/m", "dB", "dB"]):
            text = f"{v:.4g} {unit}" if np.isfinite(v) else f"∞ {unit}"
            widget.setText(text)
            widget.setToolTip(text)
        self.condition.setText(f"{snap.design_hz / 1e9:g} GHz   ·   {snap.length_m * 1000:g} mm")
        conditions = []
        if any("wavelength" in w for w in r.warnings):
            conditions.append("단면이 파장의 1/10 초과 · 전파 해석 검증 필요")
        if any("small-loss" in w for w in r.warnings):
            conditions.append("Df > 0.1 · 유전체 손실 근사 정확도 저하")
        if any("Extreme dimension" in w for w in r.warnings):
            conditions.append("큰 단면 치수비 · 독립 해석 검증 필요")
        self.set_status("   ·   ".join([KINDS[snap.geometry.kind], f"선폭 {snap.geometry.width_um:.5g} µm",
                        f"{snap.reference_ohm:g} Ω 포트", *conditions, "계산 완료"]), bool(conditions))

    def calculation_finished(self):
        self.job.deleteLater()
        self.job = None
        self.sidebar.setEnabled(True)
        self.set_exports(self.snapshot is not None and not self.dirty)

    def export(self, report, section=False):
        if self.dirty or self.snapshot is None:
            return
        metric = "section" if section else self.metric_selector.currentData()
        ext, name = ("pptx", "report") if report else ("png", metric)
        filename = f"Z-Calculator-{self.snapshot.geometry.kind}-{name}.{ext}"
        directory = self.settings.value("exports/directory", str(Path.home() / "Documents"), type=str)
        chosen, _ = QFileDialog.getSaveFileName(self, "PowerPoint 보고서 저장" if report else "PNG 저장",
                                                str(Path(directory) / filename), f"{ext.upper()} (*.{ext})")
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() != "." + ext:
            path = Path(str(path) + "." + ext)
            if path.exists() and QMessageBox.question(self, "기존 파일 교체", f"{path.name} 파일을 교체할까요?") != QMessageBox.StandardButton.Yes:
                return
        try:
            if report:
                save_pptx(path, self.snapshot)
            else:
                save_png(path, self.snapshot, metric)
            self.settings.setValue("exports/directory", str(path.parent))
            self.set_status(f"저장 완료   ·   {path}")
        except Exception as exc:
            QMessageBox.warning(self, "저장 실패", str(exc))

    def resizeEvent(self, event):
        if hasattr(self, "subtitle"):
            self.subtitle.setVisible(event.size().width() > 850)
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.job is not None and self.job.isRunning():
            self.set_status("계산이 끝나면 창을 자동으로 닫습니다.")
            self.job.finished.connect(self.close)
            event.ignore()
        else:
            event.accept()
