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

from . import __version__
from .engine import C0, Geometry, analyze, synthesize_width
from .materials import load_cmx, manual_material
from .touchstone import save_touchstone
from .visuals import (KINDS, METRICS, WARNINGS, PlotWidget, Section, SectionWidget, Snapshot,
                      frequency_text, save_png, save_pptx)


# Field key: (accepts(value), what to enter). The ranges match the engine limits.
DIMENSION = (lambda v: 0.001 <= v <= 1e7, "0.001–10,000,000 µm 범위의 값")
POSITIVE = (lambda v: v > 0, "0보다 큰 값")
FREQUENCY = (lambda v: 1e-12 <= v <= 1e6, "0보다 크고 1,000,000 GHz 이하인 값")
RULES = {
    "width_um": DIMENSION, "thickness_um": DIMENSION, "gap_um": DIMENSION,
    "bottom_um": DIMENSION, "top_um": DIMENSION, "clearance_um": DIMENSION,
    "angle_deg": (lambda v: 45 <= v <= 90, "45–90° 범위의 값"),
    "target": POSITIVE, "reference": POSITIVE,
    "start": FREQUENCY, "stop": FREQUENCY, "design": FREQUENCY,
    "length": (lambda v: v >= 0, "0 이상의 값"),
}
DK = (lambda v: 1 <= v <= 100, "1–100 범위의 값")
DF = (lambda v: 0 <= v <= 1, "0–1 범위의 값")
MANUAL = "직접 입력"


class InputError(ValueError):
    """Unusable text in one input widget; carries the widget to highlight and focus."""

    def __init__(self, widget, message):
        super().__init__(message)
        self.widget = widget


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
        QLabel#value[stale="true"], QLabel#primaryValue[stale="true"] { color: #7d8794; }
        QLabel#staleNote { color: #f0c27a; }
        QLabel#status { color: #b9c7d9; }
        QLabel#status[error="true"] { color: #ffb6ab; }
        QLineEdit, QComboBox { background: #1d2025; border: 1px solid #525b67;
            border-radius: 4px; padding: 3px 7px; min-height: 21px; }
        QLineEdit:focus, QComboBox:focus, QPushButton:focus, QCheckBox:focus {
            border: 1px solid #8fc1ff; }
        QLineEdit[invalid="true"] { border: 1px solid #ff8f80; }
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


def number(edit, rule=None):
    """Read a field as a finite number that satisfies rule = (accepts, what to enter)."""
    accepts, expected = rule or (lambda v: True, "유한한 값")
    try:
        value = float(edit.text().strip())
    except ValueError:
        value = float("nan")
    if not np.isfinite(value) or not accepts(value):
        raise InputError(edit, f"{edit.accessibleName()}: {expected}을 입력하세요.")
    return value


def flag(widget, name, on):
    """Set a boolean style property and restyle only when it changes."""
    if bool(widget.property(name)) != on:
        widget.setProperty(name, on)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def form():
    layout = QFormLayout()
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setVerticalSpacing(6)
    layout.setHorizontalSpacing(12)
    return layout


class MaterialEditor(QWidget):
    """Library selector whose Dk and Df fields always stay editable.

    Picking a library entry fills the fields with its values at the design
    frequency. Changing either field switches to manual entry, so a library
    value is a starting point rather than a lock.
    """
    changed = Signal()

    def __init__(self, title):
        super().__init__()
        self.library = {}
        self.design_hz = 1e9
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(label(title, "sectionTitle"))
        self.selector = QComboBox()
        self.selector.setAccessibleName(title)
        self.selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.selector.addItem(MANUAL, None)
        layout.addWidget(self.selector)
        row = QHBoxLayout()
        self.er, self.df = field("4.2", title + " Dk"), field("0.018", title + " Df")
        for name, edit in [("Dk", self.er), ("Df", self.df)]:
            row.addWidget(label(name, "subtle"))
            row.addWidget(edit, 1)
            edit.textChanged.connect(self._edited)
        layout.addLayout(row)
        self.details = label("", "subtle")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.selector.currentIndexChanged.connect(self._selected)
        self._describe()

    def material(self):
        name = self.selector.currentData()
        return self.library[name] if name else None

    def set_library(self, library):
        previous = self.selector.currentData()
        self.library = library
        blocked = self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItem(MANUAL, None)
        for name in sorted(library, key=str.casefold):
            self.selector.addItem(name, name)
        self.selector.setCurrentIndex(max(0, self.selector.findData(previous)) if previous else 0)
        self.selector.blockSignals(blocked)
        self._selected()

    def set_design(self, design_hz):
        if design_hz != self.design_hz:
            self.design_hz = design_hz
            self._fill()

    def _fill(self):
        """Show the selected library entry at the design frequency without leaving it."""
        material = self.material()
        if material is not None:
            er, df = material.sample(np.array([self.design_hz]))
            for edit, text in ((self.er, f"{er[0]:.5g}"), (self.df, f"{df[0]:.4g}")):
                blocked = edit.blockSignals(True)
                edit.setText(text)
                edit.blockSignals(blocked)
        self._describe()

    def _selected(self):
        self._fill()
        self.changed.emit()

    def _edited(self):
        if self.selector.currentIndex() != 0:
            blocked = self.selector.blockSignals(True)
            self.selector.setCurrentIndex(0)
            self.selector.blockSignals(blocked)
            self._describe()
        self.changed.emit()

    def _describe(self):
        material = self.material()
        if material is None:
            self.details.setText("입력한 값을 주파수 전체에 상수로 적용합니다.")
            return
        lo, hi = material.frequency_range_hz
        table = (f"{frequency_text(lo)} 단일 기준점 · 상수 적용" if material.is_constant else
                 f"{frequency_text(lo)}–{frequency_text(hi)} 표 보간 · 범위 밖 끝값")
        temperature = "온도 미지정 기본 모델" if material.temperature_c is None else f"{material.temperature_c:g} °C 모델"
        self.details.setText(f"라이브러리 값 · {frequency_text(self.design_hz)} 기준 표시\n{table}\n{temperature}\n"
                             "값을 고치면 직접 입력으로 바뀝니다.")

    def mark(self):
        """Highlight unusable Dk/Df text; return True when both are usable."""
        usable = True
        for edit, rule in ((self.er, DK), (self.df, DF)):
            try:
                number(edit, rule)
                flag(edit, "invalid", False)
            except InputError:
                flag(edit, "invalid", True)
                usable = False
        return usable

    def preview(self):
        """Material name and Dk at the design frequency (None while the Dk text is unusable)."""
        try:
            er = number(self.er, DK)
        except InputError:
            er = None
        return self.selector.currentData() or MANUAL, er

    def sample(self, frequencies):
        material = self.material() or manual_material(number(self.er, DK), number(self.df, DF))
        er, df = material.sample(frequencies)
        return self.selector.currentData() or MANUAL, er, df


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
        self.section_ok = False
        self.editing = None
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
        self.touchstone_button = QPushButton("Touchstone")
        self.touchstone_button.setToolTip("S파라미터를 Touchstone 파일로 저장합니다. 단일 선로는 .s2p, 차동쌍은 단일 종단 4포트 .s4p입니다.")
        head.addWidget(self.touchstone_button)
        self.touchstone_button.clicked.connect(self.export_touchstone)
        self.report_button = QPushButton("PowerPoint 보고서")
        self.report_button.setToolTip("조건·단면·전체 그래프·수치·모델 가정을 담은 .pptx를 저장합니다.")
        head.addWidget(self.report_button)
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
        # Last usable value of every field: the section drawing survives half-typed input.
        self.last = {key: float(edit.text()) for key, edit in self.fields.items()}
        self.kind.currentIndexChanged.connect(self.kind_changed)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.shield.toggled.connect(self.geometry_visibility)
        for edit in self.fields.values():
            edit.textChanged.connect(self.mark_dirty)
            edit.returnPressed.connect(self.calculate)
        for editor in (self.bottom, self.top):
            editor.changed.connect(self.mark_dirty)
            for edit in (editor.er, editor.df):
                edit.returnPressed.connect(self.calculate)
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
        self.mode = QComboBox()
        self.mode.setAccessibleName("계산 방식")
        self.mode.addItems(["치수 → 임피던스", "목표 임피던스 → 선폭"])
        choices = form()
        choices.addRow("구조", self.kind)
        choices.addRow("계산 방식", self.mode)
        stack_layout.addLayout(choices)
        stack_layout.addSpacing(4)
        stack_layout.addWidget(label("도체 치수", "sectionTitle"))
        self.geo_form = form()
        stack_layout.addLayout(self.geo_form)
        for key, title, value in [
            ("target", "목표 임피던스 (Ω)", 50), ("width_um", "하부 선폭 W (µm)", 180),
            ("thickness_um", "구리 두께 T (µm)", 18), ("angle_deg", "측벽 각도 θ (°)", 90),
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
        self.library_info = label("CMX 없이도 아래에 Dk·Df를 직접 입력할 수 있습니다.", "subtle")
        self.library_info.setWordWrap(True)
        mat_layout.addWidget(self.library_info)
        self.bottom, self.top = MaterialEditor("하부 유전체"), MaterialEditor("상부 유전체")
        mat_layout.addSpacing(5)
        mat_layout.addWidget(self.bottom)
        mat_layout.addSpacing(5)
        mat_layout.addWidget(self.top)
        self.air_note = label("상부 유전체는 스트립라인에서만 사용합니다. 마이크로스트립의 도선 위는 공기입니다.", "subtle")
        self.air_note.setWordWrap(True)
        mat_layout.addWidget(self.air_note)
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
        self.stale_note = label("입력 변경됨 · 아래는 이전 계산 결과", "staleNote")
        self.stale_note.hide()
        heading.addWidget(self.stale_note)
        heading.addStretch()
        self.condition = label("", "subtle")
        heading.addWidget(self.condition)
        layout.addLayout(heading)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.stat_labels, self.stat_values = [], []
        for i, caption in enumerate(["특성 임피던스", "도선 DCR (1가닥)", "반사 S11", "전송 S21"]):
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
                self.section_save.setToolTip("지금 보이는 단면 개념도를 PNG로 저장합니다.")
                self.section_save.clicked.connect(lambda: self.export(False, section=True))
                header.addWidget(self.section_save)
            else:
                self.log_axis = QCheckBox("로그 축")
                self.log_axis.setToolTip("주파수 축을 로그 눈금으로 표시합니다. 그래프 항목마다 따로 기억합니다.")
                self.log_axis.toggled.connect(self.plot.set_log_x)
                header.addWidget(self.log_axis)
                self.metric_selector = QComboBox()
                self.metric_selector.setAccessibleName("그래프 항목")
                for key, (name, _, _) in METRICS.items():
                    self.metric_selector.addItem(name, key)
                self.metric_selector.currentIndexChanged.connect(self.metric_changed)
                header.addWidget(self.metric_selector)
                self.png_button = QPushButton("그래프 PNG")
                self.png_button.setToolTip("선택한 그래프를 PNG로 저장합니다.")
                self.png_button.clicked.connect(lambda: self.export(False))
                header.addWidget(self.png_button)
            col.addLayout(header)
            col.addWidget(widget, 1)
            split.addWidget(pane)
        split.setSizes([255, 350])
        self.metric_changed()
        return right

    def metric_changed(self):
        key = self.metric_selector.currentData()
        self.plot.set_metric(key)
        blocked = self.log_axis.blockSignals(True)
        self.log_axis.setChecked(self.plot.log_x[key])
        self.log_axis.blockSignals(blocked)

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
        self.air_note.setVisible(not strip)
        self.geometry_note.setText("상·하 기준면 거리는 도선 표면에서 측정합니다." if strip else
                                   "도선 위는 공기입니다. H₁은 하부 접지면까지의 거리입니다.")
        self.port_note.setText(("차동 포트" if pair else "단일 종단 포트") +
                               " 기준입니다. S11·S21은 지정한 길이와 양단 임피던스를 사용합니다.")
        self.mark_dirty()

    def mode_changed(self):
        inverse = self.mode.currentIndex() == 1
        self.geo_form.setRowVisible(self.fields["target"], inverse)
        self.fields["width_um"].setEnabled(not inverse)
        self.fields["width_um"].setToolTip("목표 임피던스에서 계산해 채웁니다." if inverse else "")
        self.run_button.setText("목표 선폭 계산" if inverse else "임피던스 계산")
        self.mark_dirty()

    def set_status(self, message, error=False):
        self.status.setText(message)
        flag(self.status, "error", error)

    def active_fields(self):
        """Keys of the fields the current structure and mode actually use."""
        kind = self.kind.currentData()
        inverse = self.mode.currentIndex() == 1
        unused = {"width_um" if inverse else "target"}
        if "coupled" not in kind:
            unused.add("gap_um")
        if "stripline" not in kind:
            unused.add("top_um")
        if not self.shield.isChecked():
            unused.add("clearance_um")
        return self.fields.keys() - unused

    def refresh_preview(self):
        """Redraw the section from the inputs as they are now, before any calculation."""
        active = self.active_fields()
        geometry_ok = True
        for key, edit in self.fields.items():
            try:
                self.last[key] = number(edit, RULES[key])
                usable = True
            except InputError:
                usable = key not in active
            flag(edit, "invalid", not usable)
            if key.endswith(("_um", "_deg")):
                geometry_ok &= usable
        v = self.last
        kind = self.kind.currentData()
        strip = "stripline" in kind
        geometry = Geometry(kind=kind, width_um=v["width_um"], thickness_um=v["thickness_um"],
                            bottom_um=v["bottom_um"], top_um=v["top_um"], gap_um=v["gap_um"],
                            angle_deg=v["angle_deg"], shield=self.shield.isChecked(),
                            clearance_um=v["clearance_um"])
        names = []
        for editor in (self.bottom, self.top):
            editor.set_design(v["design"] * 1e9)
            if editor is self.bottom or strip:
                editor.mark()
            names.append(editor.preview())
        (bottom_name, bottom_er), (top_name, top_er) = names
        inverse = self.mode.currentIndex() == 1
        self.section.set_section(Section(geometry, bottom_name, top_name if strip else "공기",
                                         bottom_er, top_er if strip else 1.0,
                                         width_known=not inverse))
        self.section_ok = geometry_ok and not inverse and geometry.top_width_um > 0

    def set_dirty(self, dirty):
        """Track whether the displayed results still belong to the current inputs."""
        self.dirty = dirty
        stale = dirty and self.snapshot is not None
        self.plot.set_stale(stale)
        self.stale_note.setVisible(stale)
        for value in self.stat_values:
            flag(value, "stale", stale)
        self.update_exports()

    def update_exports(self):
        idle = self.job is None
        fresh = idle and self.snapshot is not None and not self.dirty
        self.png_button.setEnabled(fresh)
        self.report_button.setEnabled(fresh)
        self.touchstone_button.setEnabled(fresh)
        self.section_save.setEnabled(idle and self.section_ok)

    def mark_dirty(self):
        self.refresh_preview()
        self.set_dirty(True)
        if self.snapshot and self.job is None:
            self.set_status("입력이 변경되었습니다. 결과를 갱신하려면 계산을 실행하세요 (Ctrl+Enter).")

    def browse_library(self):
        previous = self.settings.value("materials/lastPath", "", type=str)
        filename, _ = QFileDialog.getOpenFileName(self, "유전체 라이브러리 열기", previous,
                                                   "Cadence 재료 라이브러리 (*.cmx);;모든 파일 (*)")
        if filename:
            self.open_library(Path(filename))

    def open_library(self, path, quiet=False):
        try:
            library = load_cmx(path)
        except (ValueError, OSError) as exc:
            self.library_info.setText(f"파일 열기 실패 · {exc}")
            if not quiet:
                QMessageBox.warning(self, "CMX 파일 열기", str(exc))
            return
        for editor in (self.bottom, self.top):
            editor.set_library(library)
        self.library_info.setText(f"{path.name}   ·   유전체 {len(library)}개\n"
                                  "목록에서 재료를 고르거나 Dk·Df를 직접 입력하세요.")
        self.library_info.setToolTip(str(path))
        self.settings.setValue("materials/lastPath", str(path.resolve()))

    def read_inputs(self):
        def value(key):
            return number(self.fields[key], RULES[key])
        kind = self.kind.currentData()
        pair, strip = "coupled" in kind, "stripline" in kind
        inverse = self.mode.currentIndex() == 1
        geometry = Geometry(kind=kind, width_um=value("width_um") if not inverse else 180.0,
                            thickness_um=value("thickness_um"), angle_deg=value("angle_deg"),
                            bottom_um=value("bottom_um"), top_um=value("top_um") if strip else 100.0,
                            gap_um=value("gap_um") if pair else 120.0, shield=self.shield.isChecked(),
                            clearance_um=value("clearance_um") if self.shield.isChecked() else 150.0)
        start, stop, design = (value(key) * 1e9 for key in ("start", "stop", "design"))
        if not start < stop:
            raise InputError(self.fields["stop"], "종료 주파수는 시작 주파수보다 커야 합니다.")
        if not start <= design <= stop:
            raise InputError(self.fields["design"], "설계 주파수를 시작과 종료 주파수 사이로 지정하세요.")
        length_m = value("length") / 1000
        reference = value("reference")
        target = value("target") if inverse else None
        # Include material breakpoints; both axes of the graphs need their own even spacing.
        points = [np.geomspace(start, stop, 321), [design]]
        max_er = 1.
        for editor in (self.bottom, self.top) if strip else (self.bottom,):
            material = editor.material()
            if material is not None:
                knots = material.frequencies_hz
                active = np.r_[start, knots[(knots > start) & (knots < stop)], stop]
                points.append(active)
                max_er = max(max_er, float(material.sample(active)[0].max()))
            else:
                max_er = max(max_er, number(editor.er, DK))
        # 32 points per wavelength of line length resolve the finite-line reflection ripple.
        intervals = (stop - start) / C0 * (32 * length_m) * np.sqrt(max_er)
        if intervals > 32000:
            raise InputError(self.fields["length"], "지정한 길이에서 반사 응답을 분해하려면 주파수 범위를 "
                             "나누어 계산하세요 (구간당 최대 32,000분할).")
        points.append(np.linspace(start, stop, max(401, int(np.ceil(intervals)) + 1)))
        freq = np.unique(np.concatenate(points))
        bn, be, bd = self.bottom.sample(freq)
        if strip:
            tn, te, td = self.top.sample(freq)
        else:
            tn, te, td = "공기", np.ones_like(freq), np.zeros_like(freq)
        return dict(geometry=geometry, freq_hz=freq, bottom_name=bn, bottom_er=be, bottom_df=bd,
                    top_name=tn, top_er=te, top_df=td, length_m=length_m,
                    reference_ohm=reference, design_hz=design), target

    def show_field(self, widget):
        """Bring an input with unusable text into view and focus it."""
        for index in range(self.tabs.count()):
            if self.tabs.widget(index).isAncestorOf(widget):
                self.tabs.setCurrentIndex(index)
        flag(widget, "invalid", True)
        widget.setFocus()
        widget.selectAll()

    def calculate(self):
        if self.job is not None:
            return
        try:
            inputs, target = self.read_inputs()
        except InputError as exc:
            self.set_status(str(exc), True)
            self.show_field(exc.widget)
            return
        except ValueError as exc:
            self.set_status(str(exc), True)
            return
        self.job = Calculation(inputs, target, self)
        self.job.ready.connect(self.display_result)
        self.job.failed.connect(lambda message: self.set_status(message, True))
        self.job.finished.connect(self.calculation_finished)
        # Disabling the inputs moves keyboard focus away; it is given back afterwards.
        self.editing = self.focusWidget()
        self.sidebar.setEnabled(False)
        self.update_exports()
        self.set_status("설계 주파수에서 목표 선폭을 찾는 중…" if target is not None else "단면 전계와 주파수 응답 계산 중…")
        self.job.start()

    def display_result(self, snap):
        self.snapshot = snap
        if snap.target_ohm is not None:
            self.fields["width_um"].setText(f"{snap.geometry.width_um:.8g}")
        self.section.set_section(snap.section)
        self.section_ok = True
        self.plot.set_snapshot(snap)
        self.set_dirty(False)
        i = int(np.argmin(abs(snap.freq_hz - snap.design_hz)))
        r = snap.result
        pair = "coupled" in snap.geometry.kind
        for caption, text in zip(self.stat_labels, ["차동 임피던스", "도선 DCR (1가닥)", "반사 Sdd11", "전송 Sdd21"] if pair
                                 else ["특성 임피던스", "도선 DCR (1가닥)", "반사 S11", "전송 S21"]):
            caption.setText(text)
        for widget, v, unit in zip(self.stat_values, [r.impedance_ohm[i], r.dcr_ohm_m, r.s11_db[i], r.s21_db[i]],
                                   ["Ω", "Ω/m", "dB", "dB"]):
            text = f"{v:.4g} {unit}"
            widget.setText(text)
            widget.setToolTip(text)
        self.condition.setText(f"{snap.design_hz / 1e9:g} GHz   ·   {snap.length_m * 1000:g} mm")
        conditions = [WARNINGS[key][0] for key in r.warnings if key in WARNINGS]
        self.set_status("   ·   ".join([KINDS[snap.geometry.kind], f"선폭 {snap.geometry.width_um:.5g} µm",
                        f"{snap.reference_ohm:g} Ω 포트", *conditions, "계산 완료"]), bool(conditions))

    def calculation_finished(self):
        self.job.deleteLater()
        self.job = None
        self.sidebar.setEnabled(True)
        if self.editing is None:
            if self.focusWidget() is not None:
                self.focusWidget().clearFocus()
        elif self.sidebar.isAncestorOf(self.editing) and self.editing.isEnabled():
            self.editing.setFocus()
        self.update_exports()

    def export(self, report, section=False):
        if section:
            if not self.section_save.isEnabled():
                return
            content, metric = self.section.section, "section"
        elif self.dirty or self.snapshot is None:
            return
        else:
            content, metric = self.snapshot, self.metric_selector.currentData()
        ext, name = ("pptx", "report") if report else ("png", metric)
        filename = f"Z-Calculator-{content.geometry.kind}-{name}.{ext}"
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
                save_pptx(path, content, self.plot.log_x, self.plot.marker_hz)
            else:
                save_png(path, content, metric, None if section else self.plot.log_x[metric],
                         None if section else self.plot.marker_hz)
            self.settings.setValue("exports/directory", str(path.parent))
            self.set_status(f"저장 완료   ·   {path}")
        except Exception as exc:
            QMessageBox.warning(self, "저장 실패", str(exc))

    def touchstone_comments(self, snap):
        """Header comments of a Touchstone file: what was calculated, in ASCII."""
        g, i = snap.geometry, int(np.argmin(abs(snap.freq_hz - snap.design_hz)))
        strip, pair = "stripline" in g.kind, "coupled" in g.kind
        name = lambda text: "manual entry" if text == MANUAL else text
        dimensions = f"W = {g.width_um:.6g} um (bottom), T = {g.thickness_um:g} um, sidewall {g.angle_deg:g} deg, H1 = {g.bottom_um:g} um"
        if strip:
            dimensions += f", H2 = {g.top_um:g} um"
        if pair:
            dimensions += f", S = {g.gap_um:g} um"
        if g.shield:
            dimensions += f", grounded shield at G = {g.clearance_um:g} um"
        materials = f"Bottom dielectric: {name(snap.bottom_name)} (Dk {snap.bottom_er[i]:.4g}, Df {snap.bottom_df[i]:.4g} at {snap.design_hz / 1e9:g} GHz)"
        materials += (f"; top dielectric: {name(snap.top_name)} (Dk {snap.top_er[i]:.4g}, Df {snap.top_df[i]:.4g})"
                      if strip else "; air above the trace")
        ports = (f"Single-ended ports of {snap.result.port_ohm:g} ohm (differential {snap.reference_ohm:g} ohm)"
                 if pair else f"Ports of {snap.result.port_ohm:g} ohm")
        return [f"Z Calculator {__version__}: {g.kind.replace('_', ' ')}, uniform line of {snap.length_m * 1000:g} mm",
                dimensions, materials, ports,
                "Quasi-TEM model: smooth copper at 20 C with skin effect, no roughness, vias, connectors or radiation"]

    def export_touchstone(self):
        if self.dirty or self.snapshot is None or self.snapshot.result.s_parameters is None:
            return
        snap = self.snapshot
        ports = snap.result.s_parameters.shape[-1]
        ext = f"s{ports}p"
        # Tools number the four ports of a pair in two ways; the save dialog offers both.
        orders = {f"Touchstone {ports}포트 (*.{ext})": "1-2"} if ports == 2 else {
            f"Touchstone 4포트 · 통과 1→2, 3→4 (*.{ext})": "1-2",
            f"Touchstone 4포트 · 통과 1→3, 2→4 (*.{ext})": "1-3"}
        directory = self.settings.value("exports/directory", str(Path.home() / "Documents"), type=str)
        previous = self.settings.value("exports/portOrder", "1-2", type=str)
        chosen, selected = QFileDialog.getSaveFileName(
            self, "Touchstone 저장", str(Path(directory) / f"Z-Calculator-{snap.geometry.kind}.{ext}"),
            ";;".join(orders), next((name for name, order in orders.items() if order == previous), ""))
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() != "." + ext:
            path = Path(str(path) + "." + ext)
            if path.exists() and QMessageBox.question(self, "기존 파일 교체", f"{path.name} 파일을 교체할까요?") != QMessageBox.StandardButton.Yes:
                return
        order = orders.get(selected, "1-2")
        try:
            save_touchstone(path, snap.freq_hz, snap.result.s_parameters, snap.result.port_ohm,
                            self.touchstone_comments(snap), order)
            self.settings.setValue("exports/directory", str(path.parent))
            if ports == 4:
                self.settings.setValue("exports/portOrder", order)
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
