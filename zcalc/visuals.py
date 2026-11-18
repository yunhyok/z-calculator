"""Qt drawings shared by the live view, PNG files and PowerPoint reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget


BACKGROUND = "#202226"
PANEL = "#2b2e34"
EDITOR = "#15181d"
TEXT = "#eceef1"
MUTED = "#b9c0ca"
BLUE = "#8fc1ff"
COPPER = "#dfaa70"
GRID = "#3d444f"

KINDS = {
    "microstrip": "마이크로스트립",
    "stripline": "스트립라인",
    "coupled_microstrip": "차동 마이크로스트립",
    "coupled_stripline": "차동 스트립라인",
}
METRICS = {
    "impedance": ("특성 임피던스", "Ω", "impedance_ohm"),
    "dcr": ("직류 저항 · 도체 1개", "Ω/m", "dcr_ohm_m"),
    "return": ("반사 손실", "dB", "return_loss_db"),
    "attenuation": ("단위 길이 감쇠", "dB/m", "attenuation_db_m"),
    "insertion": ("유한 길이 삽입 손실", "dB", "insertion_loss_db"),
}


@dataclass
class Snapshot:
    geometry: Any
    result: Any
    freq_hz: np.ndarray
    bottom_name: str
    top_name: str
    bottom_er: np.ndarray
    bottom_df: np.ndarray
    top_er: np.ndarray
    top_df: np.ndarray
    length_m: float
    reference_ohm: float
    design_hz: float
    target_ohm: float | None = None

    def __post_init__(self):
        self.freq_hz = np.asarray(self.freq_hz, dtype=float)
        if (self.freq_hz.ndim != 1 or self.freq_hz.size < 2
                or not np.all(np.isfinite(self.freq_hz))
                or np.any(self.freq_hz <= 0) or np.any(np.diff(self.freq_hz) <= 0)):
            raise ValueError("주파수는 양수이며 오름차순인 2개 이상의 값이어야 합니다.")
        if self.geometry.kind not in KINDS:
            raise ValueError("지원하지 않는 전송선 구조입니다.")
        if not self.freq_hz[0] <= self.design_hz <= self.freq_hz[-1]:
            raise ValueError("설계 주파수는 계산 범위 안에 있어야 합니다.")
        if not math.isfinite(self.length_m) or self.length_m < 0:
            raise ValueError("선로 길이는 0 이상의 유한한 값이어야 합니다.")
        if not math.isfinite(self.reference_ohm) or self.reference_ohm <= 0:
            raise ValueError("기준 임피던스는 양수여야 합니다.")
        for key in ("bottom_er", "bottom_df", "top_er", "top_df"):
            values = np.asarray(getattr(self, key), dtype=float)
            if values.ndim == 0:
                values = np.full(self.freq_hz.shape, float(values))
            if values.shape != self.freq_hz.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{key}: 주파수별 유한한 재료 데이터가 필요합니다.")
            setattr(self, key, values)


def frequency_text(value: float) -> str:
    for divisor, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if value >= divisor:
            return f"{value / divisor:g} {unit}"
    return f"{value:g} Hz"


def _number(value: float, precision: int = 4) -> str:
    if np.isnan(value):
        return "계산 불가"
    if np.isposinf(value):
        return "∞"
    if np.isneginf(value):
        return "−∞"
    return f"{value:.{precision}g}"


def _values(snapshot: Snapshot, metric: str) -> np.ndarray:
    if metric not in METRICS:
        raise ValueError(f"지원하지 않는 그래프: {metric}")
    values = np.asarray(getattr(snapshot.result, METRICS[metric][2]))
    if np.iscomplexobj(values):
        values = values.real
    values = values.astype(float)
    if values.ndim == 0:
        values = np.full(snapshot.freq_hz.shape, float(values))
    if values.shape != snapshot.freq_hz.shape:
        raise ValueError(f"{metric}: 계산 결과와 주파수 개수가 다릅니다.")
    return values


def _at_design(snapshot: Snapshot, values: np.ndarray) -> float:
    return float(np.interp(math.log10(snapshot.design_hz), np.log10(snapshot.freq_hz), values))


def _font(size: float, bold: bool = False) -> QFont:
    font = QFont("Malgun Gothic")
    font.setPointSizeF(size)
    font.setBold(bold)
    return font


def _text(p: QPainter, x: float, y: float, width: float, height: float,
          text: str, size: float = 12, color: str = TEXT, bold: bool = False,
          align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter):
    p.setFont(_font(size, bold))
    p.setPen(QColor(color))
    if "\n" not in text:
        text = QFontMetricsF(p.font()).elidedText(text, Qt.TextElideMode.ElideRight, width)
    p.drawText(QRectF(x, y, width, height), align, text)


def _line(p: QPainter, x1, y1, x2, y2, color=MUTED, width=1.0, dashed=False):
    pen = QPen(QColor(color), width)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawLine(QLineF(x1, y1, x2, y2))


def _dimension(p: QPainter, x1, y1, x2, y2, label: str, *, vertical=False):
    _line(p, x1, y1, x2, y2, BLUE, 1.2)
    if vertical:
        for y in (y1, y2):
            _line(p, x1 - 5, y, x1 + 5, y, BLUE, 1.2)
        _text(p, x1 - 112, min(y1, y2), 102, abs(y2 - y1), label, 11, BLUE,
              align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    else:
        for x in (x1, x2):
            _line(p, x, y1 - 5, x, y1 + 5, BLUE, 1.2)
        _text(p, min(x1, x2) - 95, y1 + 7, abs(x2 - x1) + 190, 24, label, 11, BLUE,
              align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)


class _DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot: Snapshot | None = None
        self.setMinimumSize(300, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        return QSize(1000, 564)

    def set_snapshot(self, snapshot: Snapshot | None):
        self.snapshot = snapshot
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        self.draw(p, self.width(), self.height())
        p.end()

    def draw(self, p: QPainter, width: float, height: float):
        p.fillRect(QRectF(0, 0, width, height), QColor(EDITOR))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.snapshot is not None and height < 460:
            self._draw_compact(p, width, height)
            return
        scale = min(width / 1100, height / 620)
        p.translate((width - 1100 * scale) / 2, (height - 620 * scale) / 2)
        p.scale(scale, scale)
        if self.snapshot is None:
            _text(p, 80, 260, 940, 70, "조건을 입력하고 계산할 것.", 20, MUTED,
                  align=Qt.AlignmentFlag.AlignCenter)
        else:
            self._draw(p)


class SectionWidget(_DrawingWidget):
    """Dimensioned conceptual section. Display axes are independently scaled."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAccessibleName("전송선 단면도")

    def _draw_compact(self, p: QPainter, width: float, height: float):
        s, g = self.snapshot, self.snapshot.geometry
        coupled = g.kind.startswith("coupled_")
        stripline = "microstrip" not in g.kind
        _text(p, 12, 5, width - 24, 26, KINDS[g.kind] + " / 단면 개념도", 11, bold=True)
        left, right, top, bottom = 79, width - 20, 57, height - 64
        height_fraction = np.clip(g.top_um / (g.top_um + g.bottom_um), .35, .65) if stripline else .44
        trace_top = top + (bottom - top - 11) * height_fraction
        trace_bottom = trace_top + 11
        extent = (2*g.width_um + g.gap_um if coupled else g.width_um)
        if g.shield:
            extent += 2*g.clearance_um + .7*g.width_um
        sx = (right-left)*.7 / extent
        total = (2*g.width_um + g.gap_um if coupled else g.width_um)*sx
        start, bw = (left+right-total)/2, g.width_um*sx
        inset = g.thickness_um / math.tan(math.radians(g.angle_deg))*sx
        p.fillRect(QRectF(left, top, right-left, bottom-top), QColor("#293848"))
        interface = (trace_top + trace_bottom)/2 if stripline else trace_bottom
        p.fillRect(QRectF(left, interface, right-left, bottom-interface), QColor("#344150"))
        if not stripline:
            p.fillRect(QRectF(left, top, right-left, trace_bottom-top), QColor("#1d252e"))
        else:
            p.fillRect(QRectF(left, top-5, right-left, 5), QColor("#79899a"))
        p.fillRect(QRectF(left, bottom, right-left, 6), QColor("#79899a"))
        positions = [start, start+(g.width_um+g.gap_um)*sx] if coupled else [start]
        for x in positions:
            p.setBrush(QColor(COPPER))
            p.setPen(QPen(QColor("#ffe2bd"), 1))
            p.drawPolygon(QPolygonF([QPointF(x, trace_bottom), QPointF(x+bw, trace_bottom),
                                    QPointF(x+bw-inset, trace_top), QPointF(x+inset, trace_top)]))
        if g.shield:
            a, b = start-g.clearance_um*sx, positions[-1]+bw+g.clearance_um*sx
            p.fillRect(QRectF(left, trace_top, a-left, 11), QColor("#79899a"))
            p.fillRect(QRectF(b, trace_top, right-b, 11), QColor("#79899a"))
            for x, end in ((left, a), (b, right)):
                _text(p, x-4, trace_top-20, end-x+8, 18, "GND", 8, MUTED,
                      align=Qt.AlignmentFlag.AlignCenter)
        _text(p, left+5, top-24, right-left-65, 21,
              f"상부 {s.top_name}" if stripline else "상부 공기", 9, MUTED)
        if stripline:
            _text(p, right-55, top-24, 55, 21, "GND", 8, MUTED,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _text(p, left+5, bottom+8, right-left-65, 20, f"하부 {s.bottom_name}", 9, MUTED)
        _text(p, right-55, bottom+8, 55, 20, "GND", 8, MUTED,
              align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Short leader lines preserve the dimension sketch even at 550 × 210.
        dimension_spans = [(trace_bottom, bottom)]
        if stripline:
            dimension_spans.append((top, trace_top))
        for y1, y2 in dimension_spans:
            _line(p, left-9, y1, left-9, y2, BLUE)
            _line(p, left-13, y1, left-5, y1, BLUE)
            _line(p, left-13, y2, left-5, y2, BLUE)
        dimension_y = trace_bottom+9
        _line(p, start, dimension_y, start+bw, dimension_y, BLUE)
        for x in (start, start+bw):
            _line(p, x, trace_bottom+3, x, dimension_y+3, BLUE)
        _text(p, start-32, dimension_y+1, bw+64, 17, f"w {g.width_um:g}", 8, BLUE,
              align=Qt.AlignmentFlag.AlignCenter)
        if coupled:
            gap_x = positions[0]+bw
            _line(p, gap_x, trace_top-5, positions[1], trace_top-5, BLUE)
            for x in (gap_x, positions[1]):
                _line(p, x, trace_top-8, x, trace_top-2, BLUE)
            _text(p, gap_x-30, trace_top-24, positions[1]-gap_x+60, 17,
                  f"s {g.gap_um:g}", 8, BLUE, align=Qt.AlignmentFlag.AlignCenter)
        _text(p, 3, trace_bottom, 60, max(20, bottom-trace_bottom), f"h₁ {g.bottom_um:g}", 9, BLUE,
              align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if stripline:
            _text(p, 3, top-4, 60, max(20, trace_top-top), f"h₂ {g.top_um:g}", 9, BLUE,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        summary = f"w {g.width_um:g}   t {g.thickness_um:g}   θ {g.angle_deg:g}°"
        if coupled:
            summary += f"   s {g.gap_um:g}"
        if g.shield:
            summary += f"   c {g.clearance_um:g}"
        _text(p, 12, height-31, width-24, 24, summary + "  µm · 축척 다름", 9, COPPER)

    def _draw(self, p: QPainter):
        s, g = self.snapshot, self.snapshot.geometry
        coupled = g.kind.startswith("coupled_")
        stripline = g.kind.endswith("stripline") and not g.kind.endswith("microstrip")
        top_width = g.width_um - 2 * g.thickness_um / math.tan(math.radians(g.angle_deg))
        _text(p, 35, 20, 1000, 42, KINDS[g.kind] + " / 단면 구조", 23, bold=True)
        _text(p, 35, 65, 1000, 28, "치수 단위 µm · 개념도 · 가로/세로 축척 다름", 11, MUTED)
        # Independent display scales keep thin copper visible; dimensions remain physical values.
        extent = (2 * g.width_um + g.gap_um if coupled else g.width_um)
        if g.shield:
            extent += 2 * (g.clearance_um + .35 * g.width_um)
        sx = 530 / extent
        conductor_y = 290
        thickness = 30
        bottom_y = 450
        upper_y = 145
        left, right = 220, 1010
        p.fillRect(QRectF(left, upper_y, right - left, bottom_y - upper_y), QColor("#293848"))
        interface = conductor_y + (thickness/2 if stripline else thickness)
        p.fillRect(QRectF(left, interface, right - left, bottom_y - interface), QColor("#344150"))
        if not stripline:
            p.fillRect(QRectF(left, upper_y, right - left, conductor_y + thickness - upper_y), QColor("#1d252e"))
        p.fillRect(QRectF(left, bottom_y, right - left, 14), QColor("#79899a"))
        _text(p, 235, bottom_y + 18, 700, 24, "하부 접지면 (GND)", 11, MUTED)
        if stripline:
            p.fillRect(QRectF(left, upper_y - 14, right - left, 14), QColor("#79899a"))
            _text(p, 235, upper_y - 42, 680, 24, "상부 접지면 (GND)", 11, MUTED)
        else:
            _text(p, 240, 155, 650, 28, "상부: 공기 (εr = 1)", 12, MUTED)
        er = _at_design(s, s.bottom_er)
        _text(p, 240, 394, 700, 28, f"하부 유전체: {s.bottom_name}   εr = {_number(er)}", 12)
        if stripline:
            _text(p, 240, 177, 740, 28,
                  f"상부 유전체: {s.top_name}   εr = {_number(_at_design(s, s.top_er))}", 12)
        total_trace = (2 * g.width_um + g.gap_um if coupled else g.width_um) * sx
        start = (left + right - total_trace) / 2
        bottom_width, upper_width = g.width_um * sx, max(0, top_width * sx)
        positions = [start, start + (g.width_um + g.gap_um) * sx] if coupled else [start]
        for i, x in enumerate(positions):
            inset = (bottom_width - upper_width) / 2
            p.setBrush(QColor(COPPER))
            p.setPen(QPen(QColor("#ffe2bd"), 1.4))
            p.drawPolygon(QPolygonF([
                QPointF(x, conductor_y + thickness), QPointF(x + bottom_width, conductor_y + thickness),
                QPointF(x + bottom_width - inset, conductor_y), QPointF(x + inset, conductor_y),
            ]))
            _text(p, x - 20, conductor_y - 35, bottom_width + 40, 26,
                  ("신호 +" if i == 0 else "신호 −") if coupled else "신호 도체", 11, COPPER,
                  align=Qt.AlignmentFlag.AlignCenter)
        _dimension(p, positions[0], 347, positions[0] + bottom_width, 347,
                   f"w(하부) = {g.width_um:g}")
        if coupled:
            inner = positions[0] + bottom_width
            _dimension(p, inner, 215, positions[1], 215, f"s = {g.gap_um:g}")
        _dimension(p, 203, conductor_y + thickness, 203, bottom_y,
                   f"h₁ = {g.bottom_um:g}", vertical=True)
        if stripline:
            _dimension(p, 203, upper_y, 203, conductor_y,
                       f"h₂ = {g.top_um:g}", vertical=True)
        if g.shield:
            shield_w = .35 * g.width_um * sx
            clearance = g.clearance_um * sx
            shield_xs = [positions[0] - clearance - shield_w,
                         positions[-1] + bottom_width + clearance]
            rail_end = shield_xs[0] + shield_w
            for x, end in ((left, rail_end), (shield_xs[1], right)):
                p.fillRect(QRectF(x, conductor_y, end - x, thickness), QColor("#79899a"))
                _text(p, x, conductor_y - 36, end - x, 25, "GND", 11, MUTED,
                      align=Qt.AlignmentFlag.AlignCenter)
            _dimension(p, shield_xs[1] - clearance, 347, shield_xs[1], 347,
                       f"c = {g.clearance_um:g}")
        _line(p, 35, 510, 1065, 510, GRID)
        _text(p, 35, 520, 1030, 28,
              f"두께 t = {g.thickness_um:g} µm     측벽각 θ = {g.angle_deg:g}°     상부 폭 = {top_width:.3g} µm",
              12, COPPER)
        description = "s: 두 도체의 하부 가장자리 간격" if coupled else "w: 도체의 하부 폭"
        if g.shield:
            description += "     c: 바깥 도체 하부 가장자리~동일 층 접지 간격"
        _text(p, 35, 559, 1030, 28, description, 11, MUTED)


def _y_ticks(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0, np.linspace(0, 1, 6)
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    pad = span * .09 if span > max(abs(low), abs(high), 1e-9) * 1e-8 else max(abs(low) * .08, .05)
    low, high = low - pad, high + pad
    if finite.min() >= 0:
        low = max(0.0, low)
    raw = (high - low) / 5
    power = 10 ** math.floor(math.log10(raw))
    step = next(x for x in (1, 2, 2.5, 5, 10) if x * power >= raw) * power
    low, high = math.floor(low / step) * step, math.ceil(high / step) * step
    return low, high, np.arange(low, high + .5 * step, step)


def _x_ticks(low: float, high: float):
    major, minor = [], []
    for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1):
        for factor in (1, 2, 5):
            value = factor * 10.0 ** exponent
            if low <= value <= high:
                (major if factor == 1 else minor).append(value)
    if len(major) < 2:
        major = sorted(set([low, high] + minor))
        minor = []
    # Tight logarithmic sweeps still need ticks at actual, not evenly spaced, frequencies.
    if len(major) < 3:
        major = sorted(set(major + [low, high]))
    if len(major) > 7:
        major = major[::math.ceil(len(major) / 7)]
    separated = []
    min_log_gap = math.log10(high / low) * .1
    for value in major:
        if not separated or math.log10(value / separated[-1]) >= min_log_gap:
            separated.append(value)
    return separated, minor


class PlotWidget(_DrawingWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metric_key = "impedance"
        self.setAccessibleName("주파수별 전송선 그래프")

    def set_metric(self, key: str):
        if key not in METRICS:
            raise ValueError(f"지원하지 않는 그래프: {key}")
        self.metric_key = key
        self.update()

    def _draw_compact(self, p: QPainter, width: float, height: float):
        s = self.snapshot
        title, unit, _ = METRICS[self.metric_key]
        if self.metric_key == "impedance" and s.geometry.kind.startswith("coupled_"):
            title = "차동 임피던스 Re(Zc)"
        elif self.metric_key == "impedance":
            title += " Re(Zc)"
        values = _values(s, self.metric_key)
        design = _at_design(s, values)
        _text(p, 12, 3, width*.62-12, 28, title, 11, bold=True)
        _text(p, width*.62, 3, width*.38-12, 28, f"{_number(design)} {unit}", 12, BLUE, True,
              align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        rect = QRectF(72, 49, width-95, height-104)
        low, high, ticks = _y_ticks(values)
        px = lambda f: rect.left() + math.log(f/s.freq_hz[0])/math.log(s.freq_hz[-1]/s.freq_hz[0])*rect.width()
        py = lambda value: rect.bottom() - (value-low)/(high-low)*rect.height()
        # A short plot keeps three horizontal labels at a readable fixed font size.
        step = max(1, math.ceil((len(ticks)-1)/3))
        for value in ticks[::step]:
            y = py(value)
            _line(p, rect.left(), y, rect.right(), y, GRID)
            _text(p, 3, y-10, 60, 20, _number(value), 8.5, MUTED,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        major, _ = _x_ticks(s.freq_hz[0], s.freq_hz[-1])
        major = major[::max(1, math.ceil(len(major)/max(3, int(width/110))))]
        for f in major:
            x = px(f)
            _line(p, x, rect.top(), x, rect.bottom(), GRID)
            _text(p, x-43, rect.bottom()+5, 86, 20, frequency_text(f), 8, MUTED,
                  align=Qt.AlignmentFlag.AlignCenter)
        _text(p, 8, 29, 58, 20, unit, 8.5, MUTED)
        p.save()
        p.setClipRect(rect.adjusted(-1, -1, 1, 1))
        path, active = QPainterPath(), False
        for f, value in zip(s.freq_hz, values):
            if not np.isfinite(value):
                active = False
                continue
            point = QPointF(px(f), py(value))
            if active:
                path.lineTo(point)
            else:
                path.moveTo(point)
            active = True
        p.setPen(QPen(QColor(BLUE), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        x = px(s.design_hz)
        _line(p, x, rect.top(), x, rect.bottom(), COPPER, 1.1, True)
        if np.isfinite(design):
            p.setBrush(QColor(COPPER))
            p.drawEllipse(QPointF(x, py(design)), 3.5, 3.5)
        p.restore()
        if not np.any(np.isfinite(values)):
            _text(p, rect.x(), rect.y(), rect.width(), rect.height(), "∞ dB · 완전 정합", 12, BLUE,
                  align=Qt.AlignmentFlag.AlignCenter)
        footer = "DC · 도체 1개" if self.metric_key == "dcr" else f"설계 {frequency_text(s.design_hz)}"
        _text(p, 12, height-25, width-24, 21, f"로그 주파수   /   {footer}", 8.5, MUTED,
              align=Qt.AlignmentFlag.AlignCenter)

    def _draw(self, p: QPainter):
        s = self.snapshot
        title, unit, _ = METRICS[self.metric_key]
        values = _values(s, self.metric_key)
        design = _at_design(s, values)
        if self.metric_key == "impedance" and s.geometry.kind.startswith("coupled_"):
            title = "차동 특성 임피던스"
        _text(p, 35, 18, 720, 43, title, 23, bold=True)
        _text(p, 755, 19, 310, 43, f"{_number(design)} {unit}", 23, BLUE, True,
              align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _text(p, 35, 65, 1030, 28,
              f"{KINDS[s.geometry.kind]}   ·   설계 {frequency_text(s.design_hz)}   ·   길이 {s.length_m * 1000:g} mm   ·   기준 {s.reference_ohm:g} Ω",
              11, MUTED)
        rect = QRectF(112, 133, 918, 355)
        p.fillRect(rect, QColor("#191d23"))
        low, high, yticks = _y_ticks(values)
        logf = np.log10(s.freq_hz)
        x0, x1 = float(logf[0]), float(logf[-1])
        px = lambda f: rect.left() + (math.log10(f) - x0) / (x1 - x0) * rect.width()
        py = lambda v: rect.bottom() - (v - low) / (high - low) * rect.height()
        for tick in yticks:
            y = py(tick)
            _line(p, rect.left(), y, rect.right(), y, GRID)
            _text(p, 8, y - 12, 91, 24, _number(tick), 10, MUTED,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        major, minor = _x_ticks(s.freq_hz[0], s.freq_hz[-1])
        for value in minor:
            _line(p, px(value), rect.top(), px(value), rect.bottom(), "#2d333d", .8, True)
        for value in major:
            x = px(value)
            _line(p, x, rect.top(), x, rect.bottom(), GRID)
            _text(p, x - 70, rect.bottom() + 10, 140, 26, frequency_text(value), 10, MUTED,
                  align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        _text(p, 25, 100, 100, 24, unit, 11, MUTED)
        _text(p, 380, 533, 420, 26, "주파수 (로그 축)", 12, MUTED,
              align=Qt.AlignmentFlag.AlignCenter)
        p.save()
        p.setClipRect(rect.adjusted(-1, -1, 1, 1))
        path, open_path = QPainterPath(), False
        for frequency, value in zip(s.freq_hz, values):
            if not np.isfinite(value):
                open_path = False
                continue
            point = QPointF(px(frequency), py(value))
            if open_path:
                path.lineTo(point)
            else:
                path.moveTo(point)
                open_path = True
        p.setPen(QPen(QColor(BLUE), 2.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        marker_x = px(s.design_hz)
        _line(p, marker_x, rect.top(), marker_x, rect.bottom(), COPPER, 1.4, True)
        if np.isfinite(design):
            p.setPen(QPen(QColor(EDITOR), 2))
            p.setBrush(QColor(COPPER))
            p.drawEllipse(QPointF(marker_x, py(design)), 5.5, 5.5)
        p.restore()
        if not np.any(np.isfinite(values)):
            _text(p, rect.x(), rect.y(), rect.width(), rect.height(),
                  "∞ dB · 완전 정합" if np.all(np.isposinf(values)) else "표시 가능한 유한한 결과 없음",
                  19, BLUE, align=Qt.AlignmentFlag.AlignCenter)
        notes = {
            "impedance": "손실 RLGC 모델의 Re(Zc) · 차동 Z = 2 × Zodd · 구리색 점선: 설계 주파수",
            "dcr": "도체 1개의 단위 길이 DC 저항 · 주파수에 무관 · 차동 루프는 도체 2개 합산",
            "return": "RL = −20 log₁₀|S₁₁| · 값이 클수록 반사 작음 · 양 끝 동일 기준 포트",
            "attenuation": "진행파의 단위 길이 손실 · 도체 및 유전체 손실 포함",
            "insertion": "IL = −20 log₁₀|S₂₁| · 실제 길이와 양 끝 부정합 포함",
        }
        _text(p, 35, 576, 1030, 27, notes[self.metric_key], 11, MUTED)


_owned_application = None


def _application():
    global _owned_application
    if QApplication.instance() is None:
        _owned_application = QApplication([])


def _image(snapshot: Snapshot, metric: str) -> QImage:
    _application()
    widget = SectionWidget() if metric == "section" else PlotWidget()
    if metric != "section":
        widget.set_metric(metric)
    widget.set_snapshot(snapshot)
    image = QImage(1760, 992, QImage.Format.Format_ARGB32)
    p = QPainter(image)
    try:
        widget.draw(p, image.width(), image.height())
    finally:
        p.end()
        widget.deleteLater()
    return image


def _atomic_path(path: str | Path):
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix,
                                    dir=destination.parent)
    os.close(handle)
    return destination, Path(name)


def save_png(path: str | Path, snapshot: Snapshot, metric: str):
    """Write one graph or the section, preserving an existing file on failure."""
    image = _image(snapshot, metric)
    destination, temporary = _atomic_path(path)
    try:
        if not image.save(str(temporary), "PNG"):
            raise OSError("PNG 파일을 저장하지 못했습니다.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _assumptions(snapshot: Snapshot) -> list[str]:
    supplied = getattr(snapshot.result, "assumptions", None)
    generic = [
        "균일한 직선 전송선의 준정적 단면 근사. 단면도는 치수 설명용이며 가로/세로 축척 다름.",
        "마이크로스트립 상부는 공기. 스트립라인은 위·아래 유전체와 두 접지면 적용.",
        "스트립라인의 도체 밖 재료 경계는 동박 두께의 중앙. 마이크로스트립 재료 경계는 동박 하부.",
        "w와 s는 도체 하부 기준. 측벽각 90°는 직사각형. 동일 층 접지는 바깥 도체에서 c만큼 이격.",
        "차동 Z는 2×Zodd = Vdiff/I. 표시 임피던스는 손실 RLGC 모델의 Re(Zc). DCR은 도체 1개 기준.",
        "RL과 IL은 입력·출력의 동일한 기준 임피던스에 연결한 유한 길이 선로의 양수 손실값.",
        "구리는 20°C 평활 표면 가정. 이상적인 무한 폭 접지의 DC 저항은 0. 차동 루프 DCR은 2배.",
        "거칠기·비아·커넥터·굽힘·방사·이방성·고차 모드 제외. 실제 설계는 필드 해석/측정으로 확인할 것.",
    ]
    warning_translations = {
        "Cross-section exceeds one tenth wavelength; higher-order/full-wave validation required.":
            "단면 치수가 파장의 1/10 초과. 고차 모드·전파 해석으로 확인할 것.",
        "Dielectric loss uses a small-loss perturbation; tan(delta)>0.1 reduces accuracy.":
            "유전체 손실은 작은 손실 근사. Df > 0.1에서는 정확도 저하.",
    }
    already_stated = {
        "2D quasi-TEM estimate; smooth copper at 20 C; no roughness, vias or radiation.",
        "DC resistance is per signal conductor; ideal infinite return planes/rails have zero DC resistance.",
        "Exposed microstrip: all top-layer inputs are ignored; upper medium is air.",
    }
    warnings = [warning_translations.get(message, message)
                for message in getattr(snapshot.result, "warnings", ())
                if message not in already_stated]
    if supplied:
        supplied = [supplied] if isinstance(supplied, str) else list(supplied)
        return list(dict.fromkeys([*supplied, *generic, *warnings]))
    return [*generic, *warnings]


def save_pptx(path: str | Path, snapshot: Snapshot):
    """Export editable conditions/results and the exact Qt figures in a report."""
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.oxml.xmlchemy import OxmlElement
    from pptx.util import Inches, Pt

    s, g = snapshot, snapshot.geometry
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    prs.core_properties.title = "Z Calculator 전송선 계산 보고서"
    prs.core_properties.subject = KINDS[g.kind]
    prs.core_properties.author = "Z Calculator"
    prs.core_properties.keywords = "impedance, transmission line, 준정적 근사"
    rgb = lambda value: RGBColor.from_string(value.lstrip("#"))

    def text(slide, x, y, w, h, content, size=18, color=TEXT, bold=False):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.clear()
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = Inches(.02)
        frame.margin_top = frame.margin_bottom = 0
        for i, line in enumerate(str(content).split("\n")):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.text = line
            para.font.name, para.font.size = "Malgun Gothic", Pt(size)
            para.font.bold, para.font.color.rgb = bold, rgb(color)
            para.space_after = Pt(10)
        return box

    def slide(title, subtitle=""):
        item = prs.slides.add_slide(prs.slide_layouts[6])
        item.background.fill.solid()
        item.background.fill.fore_color.rgb = rgb(BACKGROUND)
        text(item, .55, .28, 12.2, .55, title, 28, bold=True)
        if subtitle:
            text(item, .58, .93, 12.1, .42, subtitle, 13, MUTED)
        text(item, .58, 7.12, 11.2, .22, "Z Calculator  /  계산 모델 기반 예측 · 실측 검증 필요", 10, MUTED)
        text(item, 12, 7.12, .75, .22, f"{len(prs.slides):02d}", 10, MUTED)
        return item

    def table(item, headers, rows, x=.58, y=1.55, w=12.15, h=4.9):
        obj = item.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
        tbl = obj.table
        for row_index, row_values in enumerate([headers, *rows]):
            for col, value in enumerate(row_values):
                cell = tbl.cell(row_index, col)
                cell.text = str(value)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb("#dce4ee" if row_index == 0 else "#ffffff")
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.margin_left = cell.margin_right = Inches(.06)
                cell.margin_top = cell.margin_bottom = Inches(.035)
                properties = cell._tc.get_or_add_tcPr()
                for position, edge in enumerate(("L", "R", "T", "B")):
                    border = OxmlElement("a:ln" + edge)
                    border.set("w", "9525")
                    fill = OxmlElement("a:solidFill")
                    color = OxmlElement("a:srgbClr")
                    color.set("val", "15181D")
                    fill.append(color)
                    border.append(fill)
                    dash = OxmlElement("a:prstDash")
                    dash.set("val", "solid")
                    border.append(dash)
                    properties.insert(position, border)
                for para in cell.text_frame.paragraphs:
                    para.alignment = PP_ALIGN.CENTER
                    para.font.name, para.font.size = "Malgun Gothic", Pt(15 if len(rows) < 9 else 12)
                    para.font.bold = row_index == 0
                    para.font.color.rgb = rgb("#15181d")
        return tbl

    def metric_row(key):
        label, unit, _ = METRICS[key]
        vals = _values(s, key)
        return [label, f"{_number(_at_design(s, vals))} {unit}"]

    mode = "치수 입력" if s.target_ohm is None else f"선폭 역산 · 목표 {s.target_ohm:g} Ω"
    item = slide("01 / 전송선 계산 결과", f"{KINDS[g.kind]}  ·  {mode}  ·  {datetime.now():%Y-%m-%d %H:%M}")
    text(item, .62, 1.52, 11.8, .7, f"설계 주파수 {frequency_text(s.design_hz)}", 32, BLUE, True)
    table(item, ["계산 항목", "설계 주파수 결과"], [metric_row(key) for key in METRICS], y=2.52, h=3.35)
    text(item, .63, 6.2, 11.9, .5,
         f"기준 {s.reference_ohm:g} Ω  ·  선로 길이 {s.length_m * 1000:g} mm  ·  {frequency_text(s.freq_hz[0])} ~ {frequency_text(s.freq_hz[-1])}",
         17, MUTED)

    item = slide("02 / 입력 조건 - 단면 치수", "w·s·c는 도체 하부 기준 · 상부 폭은 측벽각으로 계산")
    rows = [["구조", KINDS[g.kind]], ["하부 폭 w / 두께 t", f"{g.width_um:g} / {g.thickness_um:g} µm"],
            ["상부 폭 / 측벽각 θ", f"{g.width_um - 2*g.thickness_um/math.tan(math.radians(g.angle_deg)):.4g} µm / {g.angle_deg:g}°"],
            ["하부 유전체 높이 h₁", f"{g.bottom_um:g} µm"],
            ["상부 유전체 높이 h₂", f"{g.top_um:g} µm" if "microstrip" not in g.kind else "해당 없음 · 상부 공기"],
            ["도체 간격 s", f"{g.gap_um:g} µm" if g.kind.startswith("coupled_") else "해당 없음"],
            ["동일 층 접지 / 간격 c", f"사용 / {g.clearance_um:g} µm" if g.shield else "사용 안 함"],
            ["길이 / 기준 임피던스", f"{s.length_m * 1000:g} mm / {s.reference_ohm:g} Ω"]]
    table(item, ["항목", "입력값"], rows)

    item = slide("03 / 재료와 주파수 - 계산 데이터", "CMX 주파수 표는 로그 주파수 보간 · 범위 밖 끝값 유지 · 단일 기준점과 수동 재료는 상수 적용")
    material_rows = []
    for side, name, ers, dfs in (("하부", s.bottom_name, s.bottom_er, s.bottom_df),
                                 ("상부", s.top_name, s.top_er, s.top_df)):
        if side == "상부" and "microstrip" in g.kind:
            material_rows.append([side, "공기", "1", "0", "상부 재료 입력 미적용"])
        else:
            material_rows.append([side, name, _number(_at_design(s, ers)), _number(_at_design(s, dfs)),
                                  f"εr {_number(ers.min())}~{_number(ers.max())}\nDf {_number(dfs.min())}~{_number(dfs.max())}"])
    table(item, ["위치", "재료", "설계 εr", "설계 Df", "계산 범위 내 변화"], material_rows, y=1.65, h=2.25)
    text(item, .65, 4.35, 11.9, 1.7,
         f"주파수 범위: {frequency_text(s.freq_hz[0])} ~ {frequency_text(s.freq_hz[-1])}\n"
         f"계산 점수: {len(s.freq_hz):,}개 / 그래프 가로축: 로그 주파수\n"
         f"설계 주파수: {frequency_text(s.design_hz)}", 20)

    with tempfile.TemporaryDirectory(prefix="zcalc-report-") as folder:
        for number, metric in enumerate(["section", *METRICS], start=4):
            label = "단면 구조 - 연결과 치수" if metric == "section" else METRICS[metric][0]
            item = slide(f"{number:02d} / {label}")
            picture = Path(folder) / f"{metric}.png"
            save_png(picture, s, metric)
            item.shapes.add_picture(str(picture), Inches(1.55), Inches(1.1), height=Inches(5.78))

        item = slide("10 / 수치 결과 - 대표 주파수", "원본 계산점 중 대표값 · 설계 주파수는 별도 보간값")
        indices = sorted(set(np.linspace(0, len(s.freq_hz) - 1, 5).astype(int)))
        rows = []
        for index in indices:
            rows.append([frequency_text(s.freq_hz[index]), *[_number(_values(s, key)[index]) for key in METRICS]])
        rows.append(["설계 " + frequency_text(s.design_hz), *[_number(_at_design(s, _values(s, key))) for key in METRICS]])
        table(item, ["주파수", "Z (Ω)", "DCR (Ω/m)", "RL (dB)", "감쇠 (dB/m)", "IL (dB)"], rows, h=4.25)
        text(item, .64, 6.15, 12, .45, "DCR: 도체 1개 기준. 차동 Z = 2 × Zodd = Vdiff/I.", 15, MUTED)

        assumptions = _assumptions(s)
        for offset in range(0, len(assumptions), 5):
            item = slide(f"{len(prs.slides)+1:02d} / 모델 가정과 결과 해석", "예측값의 적용 범위와 설계 확인 항목")
            for index, sentence in enumerate(assumptions[offset:offset + 5]):
                text(item, .68, 1.52 + index * .94, .45, .55, f"{offset + index + 1:02d}", 17, BLUE, True)
                important = "확인할 것" in sentence or "정확도 저하" in sentence
                text(item, 1.28, 1.48 + index * .94, 11.25, .82, sentence, 18,
                     "#ff9b9b" if important else TEXT, important)

        destination, temporary = _atomic_path(path)
        try:
            prs.save(str(temporary))
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
