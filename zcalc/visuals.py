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
from PySide6.QtCore import QLineF, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget

from zcalc.engine import COPPER_CONDUCTIVITY_S_M


BACKGROUND = "#202226"
PANEL = "#2b2e34"
EDITOR = "#15181d"
TEXT = "#eceef1"
MUTED = "#b9c0ca"
BLUE = "#8fc1ff"
STALE = "#5d7694"
AMBER = "#f0c27a"
MARKER = "#8fe3b0"
COPPER = "#dfaa70"
GRID = "#3d444f"

KINDS = {
    "microstrip": "마이크로스트립",
    "stripline": "스트립라인",
    "coupled_microstrip": "차동 마이크로스트립",
    "coupled_stripline": "차동 스트립라인",
}
# key: (title, unit, Result attribute). The titles are also the graph selector entries.
METRICS = {
    "impedance": ("특성 임피던스", "Ω", "impedance_ohm"),
    "dcr": ("단위 길이당 DCR", "Ω/m", "dcr_ohm_m"),
    "s11": ("반사 S11", "dB", "s11_db"),
    "attenuation": ("단위 길이당 전파 손실", "dB/m", "attenuation_db_m"),
    "s21": ("전송 S21", "dB", "s21_db"),
}
# Default frequency axis: S-parameters and loss read best on a linear axis.
LOG_X = {"impedance": True, "dcr": True, "s11": False, "attenuation": False, "s21": False}
# engine.WARNINGS key: (status-bar phrase, report sentence)
WARNINGS = {
    "wavelength": ("단면이 파장의 1/10 초과 · 전파 해석 검증 필요",
                   "단면 치수가 파장의 1/10 초과. 고차 모드·전파 해석으로 확인할 것."),
    "loss_tangent": ("Df > 0.1 · 유전체 손실 근사 정확도 저하",
                     "유전체 손실은 작은 손실 근사. Df > 0.1에서는 정확도 저하."),
    "aspect_ratio": ("단면 치수비 100:1 초과 · 독립 해석 검증 필요",
                     "단면 치수비가 100:1 초과. 격자 정확도를 독립 해석으로 확인할 것."),
}


@dataclass
class Section:
    """What the section drawing shows; available before any calculation."""
    geometry: Any
    bottom_name: str
    top_name: str
    bottom_er: float | None = None  # Dk at the design frequency, None if not entered yet
    top_er: float | None = None
    width_known: bool = True  # False while the width still has to be synthesized
    conductor_name: str = "구리"
    conductivity_s_m: float | None = COPPER_CONDUCTIVITY_S_M
    conductor_temperature_c: float | None = 20.0


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
    conductor_name: str = "구리"
    conductivity_s_m: float = COPPER_CONDUCTIVITY_S_M
    conductor_temperature_c: float | None = 20.0

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
        if not math.isfinite(self.conductivity_s_m) or self.conductivity_s_m <= 0:
            raise ValueError("도체 전도율은 양수인 유한한 값이어야 합니다.")
        if self.conductor_temperature_c is not None and not math.isfinite(self.conductor_temperature_c):
            raise ValueError("전도율 기준 온도는 유한한 값이어야 합니다.")
        for key in ("bottom_er", "bottom_df", "top_er", "top_df"):
            values = np.asarray(getattr(self, key), dtype=float)
            if values.ndim == 0:
                values = np.full(self.freq_hz.shape, float(values))
            if values.shape != self.freq_hz.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{key}: 주파수별 유한한 재료 데이터가 필요합니다.")
            setattr(self, key, values)

    @property
    def section(self) -> Section:
        return Section(self.geometry, self.bottom_name, self.top_name,
                       _at_design(self, self.bottom_er), _at_design(self, self.top_er),
                       conductor_name=self.conductor_name, conductivity_s_m=self.conductivity_s_m,
                       conductor_temperature_c=self.conductor_temperature_c)


def frequency_text(value: float, digits: int | None = None) -> str:
    """Frequency with its unit; digits limits the significant digits (e.g. a dragged marker)."""
    form = "g" if digits is None else f".{digits}g"
    for divisor, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if value >= divisor:
            return f"{value / divisor:{form}} {unit}"
    return f"{value:{form}} Hz"


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


def _title(snapshot: Snapshot, metric: str) -> str:
    pair = snapshot.geometry.kind.startswith("coupled_")
    if metric == "impedance":
        return "차동 임피던스 Re(Zc)" if pair else "특성 임피던스 Re(Zc)"
    if pair and metric in ("s11", "s21"):
        return {"s11": "차동 반사 Sdd11", "s21": "차동 전송 Sdd21"}[metric]
    return METRICS[metric][0]


def _at(snapshot: Snapshot, values: np.ndarray, frequency_hz: float) -> float:
    """Value of a curve at any frequency inside the sweep."""
    return float(np.interp(frequency_hz, snapshot.freq_hz, values))


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
    placeholder = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.content = None
        self.setMinimumSize(300, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        return QSize(1000, 564)

    def set_content(self, content):
        self.content = content
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        self.draw(p, self.width(), self.height())
        p.end()

    def draw(self, p: QPainter, width: float, height: float):
        p.fillRect(QRectF(0, 0, width, height), QColor(EDITOR))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.content is None:
            _text(p, 12, 0, width - 24, height, self.placeholder, 11, MUTED,
                  align=Qt.AlignmentFlag.AlignCenter)
        elif height < 460:
            self._draw_compact(p, width, height)
        else:
            scale = min(width / 1100, height / 620)
            p.translate((width - 1100 * scale) / 2, (height - 620 * scale) / 2)
            p.scale(scale, scale)
            self._draw(p)


def _material_label(side: str, name: str, er: float | None) -> str:
    return f"{side} {name}" + ("" if er is None else f" · Dk {_number(er)}")


def _conductor_label(content: Snapshot | Section) -> str:
    conductivity = ("입력 필요" if content.conductivity_s_m is None
                    else f"{content.conductivity_s_m:.6g} S/m")
    temperature = ("미지정" if content.conductor_temperature_c is None
                   else f"{content.conductor_temperature_c:g}°C")
    return f"도체: {content.conductor_name} · σ = {conductivity} · 기준 온도 {temperature}"


class SectionWidget(_DrawingWidget):
    """Dimensioned conceptual section. Display axes are independently scaled."""

    placeholder = "단면 치수를 입력하면 개념도가 표시됩니다."

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAccessibleName("전송선 단면도")

    def set_section(self, section: Section | None):
        self.set_content(section)

    @property
    def section(self) -> Section | None:
        return self.content

    def _draw_compact(self, p: QPainter, width: float, height: float):
        s, g = self.content, self.content.geometry
        coupled = g.kind.startswith("coupled_")
        stripline = "microstrip" not in g.kind
        width_text = f"W {g.width_um:g}" if s.width_known else "W 역산"
        _text(p, 12, 5, width - 24, 26, f"{KINDS[g.kind]} · 도체 {s.conductor_name}", 11, bold=True)
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
        inset = min(g.thickness_um / math.tan(math.radians(g.angle_deg))*sx, bw/2)
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
              _material_label("상부", s.top_name, s.top_er) if stripline else "상부 공기 · Dk 1", 9, MUTED)
        if stripline:
            _text(p, right-55, top-24, 55, 21, "GND", 8, MUTED,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _text(p, left+5, bottom+8, right-left-65, 20,
              _material_label("하부", s.bottom_name, s.bottom_er), 9, MUTED)
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
        _text(p, start-32, dimension_y+1, bw+64, 17, width_text, 8, BLUE,
              align=Qt.AlignmentFlag.AlignCenter)
        spans = []
        if coupled:
            spans.append((positions[0]+bw, positions[1], f"S {g.gap_um:g}"))
        if g.shield:
            edge = positions[-1]+bw
            spans.append((edge, edge+g.clearance_um*sx, f"G {g.clearance_um:g}"))
        for x1, x2, text in spans:
            _line(p, x1, trace_top-5, x2, trace_top-5, BLUE)
            for x in (x1, x2):
                _line(p, x, trace_top-8, x, trace_top-2, BLUE)
            _text(p, x1-30, trace_top-24, x2-x1+60, 17, text, 8, BLUE,
                  align=Qt.AlignmentFlag.AlignCenter)
        _text(p, 3, trace_bottom, 60, max(20, bottom-trace_bottom), f"H₁ {g.bottom_um:g}", 9, BLUE,
              align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if stripline:
            _text(p, 3, top-4, 60, max(20, trace_top-top), f"H₂ {g.top_um:g}", 9, BLUE,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        summary = f"{width_text}   T {g.thickness_um:g}   θ {g.angle_deg:g}°"
        if coupled:
            summary += f"   S {g.gap_um:g}"
        if g.shield:
            summary += f"   G {g.clearance_um:g}"
        _text(p, 12, height-31, width-24, 24, summary + "   µm · 가로/세로 축척 다름", 9, COPPER)

    def _draw(self, p: QPainter):
        s, g = self.content, self.content.geometry
        coupled = g.kind.startswith("coupled_")
        stripline = g.kind.endswith("stripline") and not g.kind.endswith("microstrip")
        top_width = g.width_um - 2 * g.thickness_um / math.tan(math.radians(g.angle_deg))
        _text(p, 35, 20, 1000, 42, KINDS[g.kind] + " · 단면 구조", 23, bold=True)
        _text(p, 35, 65, 1000, 28, "치수 단위 µm · 개념도 · 가로/세로 축척 다름", 11, MUTED)
        # Independent display scales keep thin conductors visible; dimensions remain physical values.
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
            _text(p, 240, 177, 740, 28, _material_label("상부 유전체:", s.top_name, s.top_er), 12)
        else:
            _text(p, 240, 155, 650, 28, "상부: 공기 · Dk 1", 12, MUTED)
        _text(p, 240, 394, 700, 28, _material_label("하부 유전체:", s.bottom_name, s.bottom_er), 12)
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
                   f"W(하부) = {g.width_um:g}" if s.width_known else "W(하부) = 역산 예정")
        if coupled:
            inner = positions[0] + bottom_width
            _dimension(p, inner, 215, positions[1], 215, f"S = {g.gap_um:g}")
        _dimension(p, 203, conductor_y + thickness, 203, bottom_y,
                   f"H₁ = {g.bottom_um:g}", vertical=True)
        if stripline:
            _dimension(p, 203, upper_y, 203, conductor_y,
                       f"H₂ = {g.top_um:g}", vertical=True)
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
                       f"G = {g.clearance_um:g}")
        _line(p, 35, 510, 1065, 510, GRID)
        _text(p, 35, 520, 1030, 28,
              f"두께 T = {g.thickness_um:g} µm     측벽 각도 θ = {g.angle_deg:g}°     "
              + (f"상부 폭 = {top_width:.3g} µm" if s.width_known else "상부 폭 = 역산 예정"),
              12, COPPER)
        description = "S: 두 도선의 하부 가장자리 간격" if coupled else "W: 도선의 하부 선폭"
        if g.shield:
            description += "     G: 바깥 도선 하부 가장자리~동일층 접지 실드 간격"
        _text(p, 35, 559, 1030, 28, description, 11, MUTED)
        _text(p, 35, 590, 1030, 24, _conductor_label(s), 10, MUTED)


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
    if finite.max() <= 0:  # S-parameters in dB never exceed 0
        high = min(0.0, high)
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


def _linear_ticks(low: float, high: float):
    """Round ticks for a linear axis; it grows to a round bound within 5% of the span."""
    span = high - low
    raw = span / 5
    power = 10 ** math.floor(math.log10(raw))
    step = next(x for x in (1, 2, 2.5, 5, 10) if x * power >= raw) * power
    start, end = math.floor(low / step + 1e-9) * step, math.ceil(high / step - 1e-9) * step
    if low - start > .05 * span:
        start = low
    if end - high > .05 * span:
        end = high
    first, last = math.ceil(start / step - 1e-9), math.floor(end / step + 1e-9)
    major = [index * step for index in range(first, last + 1)]
    minor = [(index + .5) * step for index in range(first - 1, last + 1)
             if start < (index + .5) * step < end]
    return start, end, major, minor, step


class _FrequencyAxis:
    """Maps frequency to 0…1 across the plot and supplies tick positions and labels."""

    def __init__(self, freq_hz: np.ndarray, log: bool):
        self.log = log
        low, high = float(freq_hz[0]), float(freq_hz[-1])
        if log:
            self.low, self.high = low, high
            self.major, self.minor = _x_ticks(low, high)
            self.labels = [frequency_text(value) for value in self.major]
            self.title = "주파수 (로그 축)"
        else:
            self.low, self.high, self.major, self.minor, step = _linear_ticks(low, high)
            divisor, unit = next(((d, u) for d, u in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"))
                                  if self.high >= d), (1.0, "Hz"))
            # Enough decimals to tell neighbouring ticks apart, however narrow the sweep.
            decimals = max(0, 1 - math.floor(math.log10(step / divisor)))
            self.labels = [f"{value / divisor:.{decimals}f}".rstrip("0").rstrip(".") if decimals
                           else f"{value / divisor:.0f}" for value in self.major]
            self.title = f"주파수 ({unit})"

    def position(self, frequency: float) -> float:
        if self.high <= self.low:  # a sweep one rounding step wide
            return .5
        if self.log:
            return math.log(frequency / self.low) / math.log(self.high / self.low)
        return (frequency - self.low) / (self.high - self.low)

    def frequency(self, position: float) -> float:
        """Inverse of position."""
        if self.log:
            return self.low * (self.high / self.low) ** position
        return self.low + position * (self.high - self.low)


class PlotWidget(_DrawingWidget):
    """Frequency response with the design-frequency marker and one movable marker.

    Clicking or dragging in the plot places the movable marker; a right click
    removes it. It keeps its frequency across graphs and calculations.
    """
    placeholder = "계산을 실행하면 주파수 응답이 표시됩니다."
    marker_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.metric_key = "impedance"
        self.log_x = dict(LOG_X)
        self.stale = False
        self.marker_hz: float | None = None
        self._plot = None  # (plot rectangle in widget coordinates, frequency axis) of the last paint
        self._dragging = False
        self.setMouseTracking(True)
        self.setToolTip("그래프를 클릭하거나 끌면 마커를 놓습니다. 오른쪽 클릭으로 지웁니다.")
        self.setAccessibleName("주파수별 전송선 그래프")

    def set_marker(self, frequency_hz: float | None):
        """Place the movable marker, or remove it with None."""
        if frequency_hz != self.marker_hz:
            self.marker_hz = frequency_hz
            self.marker_changed.emit()
            self.update()

    def _marker_at(self, x: float) -> float | None:
        """Frequency under a horizontal widget position, limited to the sweep."""
        if self._plot is None or self.content is None:
            return None
        rect, axis = self._plot
        position = min(1.0, max(0.0, (x - rect.left()) / rect.width()))
        freq = self.content.freq_hz
        return float(min(freq[-1], max(freq[0], axis.frequency(position))))

    def _over_plot(self, point) -> bool:
        return self._plot is not None and self.content is not None and self._plot[0].adjusted(-4, 0, 4, 0).contains(point)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.set_marker(None)
        elif event.button() == Qt.MouseButton.LeftButton and self._over_plot(event.position()):
            self._dragging = True
            self.set_marker(self._marker_at(event.position().x()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.set_marker(self._marker_at(event.position().x()))
        self.setCursor(Qt.CursorShape.SplitHCursor if self._dragging or self._over_plot(event.position())
                       else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)

    def _draw_marker(self, p: QPainter, rect: QRectF, values: np.ndarray, px, py, unit: str,
                     size: float, radius: float):
        """Movable marker: line, point on the curve and a frequency/value readout."""
        s = self.content
        if self.marker_hz is None or not s.freq_hz[0] <= self.marker_hz <= s.freq_hz[-1]:
            return
        x, value = px(self.marker_hz), _at(s, values, self.marker_hz)
        _line(p, x, rect.top(), x, rect.bottom(), MARKER, 1.2 * radius / 3.5)
        if np.isfinite(value):
            p.setPen(QPen(QColor(EDITOR), radius / 3))
            p.setBrush(QColor(MARKER))
            p.drawEllipse(QPointF(x, min(rect.bottom(), max(rect.top(), py(value)))), radius, radius)
        text = f"{frequency_text(self.marker_hz, 4)}   {_number(value)} {unit}"
        p.setFont(_font(size, True))
        width = QFontMetricsF(p.font()).horizontalAdvance(text) + 14
        height = size * 2.2
        left = x + 7 if x + 7 + width <= rect.right() else x - 7 - width
        box = QRectF(left, rect.top() + 4, width, height)
        p.setPen(QPen(QColor(MARKER), 1))
        p.setBrush(QColor(EDITOR))
        p.drawRoundedRect(box, 3, 3)
        p.setPen(QColor(MARKER))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def set_snapshot(self, snapshot: Snapshot | None):
        self.set_content(snapshot)

    def set_metric(self, key: str):
        if key not in METRICS:
            raise ValueError(f"지원하지 않는 그래프: {key}")
        self.metric_key = key
        self.update()

    def set_log_x(self, log: bool):
        """Choose the frequency axis of the current metric; each metric remembers its own."""
        self.log_x[self.metric_key] = bool(log)
        self.update()

    def set_stale(self, stale: bool):
        """Mark the curve as belonging to inputs that have since changed."""
        if stale != self.stale:
            self.stale = stale
            self.update()

    def _curve(self, s: Snapshot, values: np.ndarray, px, py) -> QPainterPath:
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
        return path

    def _draw_compact(self, p: QPainter, width: float, height: float):
        s = self.content
        unit = METRICS[self.metric_key][1]
        values = _values(s, self.metric_key)
        design = _at_design(s, values)
        accent = STALE if self.stale else BLUE
        _text(p, 12, 3, width*.62-12, 28, _title(s, self.metric_key), 11, bold=True)
        _text(p, width*.62, 3, width*.38-12, 28, f"{_number(design)} {unit}", 12, accent, True,
              align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        rect = QRectF(72, 49, width-95, height-104)
        low, high, ticks = _y_ticks(values)
        axis = _FrequencyAxis(s.freq_hz, self.log_x[self.metric_key])
        self._plot = (QRectF(rect), axis)
        px = lambda f: rect.left() + axis.position(f)*rect.width()
        py = lambda value: rect.bottom() - (value-low)/(high-low)*rect.height()
        # A short plot keeps three horizontal labels at a readable fixed font size.
        step = max(1, math.ceil((len(ticks)-1)/3))
        for value in ticks[::step]:
            y = py(value)
            _line(p, rect.left(), y, rect.right(), y, GRID)
            _text(p, 3, y-10, 60, 20, _number(value), 8.5, MUTED,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        every = max(1, math.ceil(len(axis.major)/max(3, int(width/110))))
        for f, text in list(zip(axis.major, axis.labels))[::every]:
            x = px(f)
            _line(p, x, rect.top(), x, rect.bottom(), GRID)
            _text(p, x-43, rect.bottom()+5, 86, 20, text, 8, MUTED,
                  align=Qt.AlignmentFlag.AlignCenter)
        _text(p, 8, 29, 58, 20, unit, 8.5, MUTED)
        _text(p, 75, 30, width-95, 18, _conductor_label(s), 8, MUTED)
        p.save()
        p.setClipRect(rect.adjusted(-1, -1, 1, 1))
        p.setPen(QPen(QColor(accent), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(self._curve(s, values, px, py))
        x = px(s.design_hz)
        _line(p, x, rect.top(), x, rect.bottom(), COPPER, 1.1, True)
        if np.isfinite(design):
            p.setBrush(QColor(COPPER))
            p.drawEllipse(QPointF(x, py(design)), 3.5, 3.5)
        self._draw_marker(p, rect, values, px, py, unit, 9, 3.5)
        p.restore()
        if not np.any(np.isfinite(values)):
            _text(p, rect.x(), rect.y(), rect.width(), rect.height(), "−∞ dB · 완전 정합", 12, BLUE,
                  align=Qt.AlignmentFlag.AlignCenter)
        if self.stale:
            _text(p, rect.x(), rect.bottom()-24, rect.width()-8, 20, "이전 결과 · 입력이 바뀌어 다시 계산 필요", 9, AMBER,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer = "DC · 도선 1가닥" if self.metric_key == "dcr" else f"설계 {frequency_text(s.design_hz)}"
        if self.marker_hz is None:
            footer += "   /   그래프를 끌면 마커"
        _text(p, 12, height-25, width-24, 21, f"{axis.title}   /   {footer}", 8.5, MUTED,
              align=Qt.AlignmentFlag.AlignCenter)

    def _draw(self, p: QPainter):
        s = self.content
        unit = METRICS[self.metric_key][1]
        values = _values(s, self.metric_key)
        design = _at_design(s, values)
        accent = STALE if self.stale else BLUE
        _text(p, 35, 18, 720, 43, _title(s, self.metric_key), 23, bold=True)
        _text(p, 755, 19, 310, 43, f"{_number(design)} {unit}", 23, accent, True,
              align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _text(p, 35, 65, 1030, 28,
              f"{KINDS[s.geometry.kind]}   ·   설계 {frequency_text(s.design_hz)}   ·   길이 {s.length_m * 1000:g} mm   ·   기준 {s.reference_ohm:g} Ω",
              11, MUTED)
        _text(p, 135, 99, 930, 24, _conductor_label(s), 10, MUTED)
        rect = QRectF(112, 133, 918, 355)
        p.fillRect(rect, QColor("#191d23"))
        low, high, yticks = _y_ticks(values)
        axis = _FrequencyAxis(s.freq_hz, self.log_x[self.metric_key])
        self._plot = (p.worldTransform().mapRect(rect), axis)
        px = lambda f: rect.left() + axis.position(f) * rect.width()
        py = lambda v: rect.bottom() - (v - low) / (high - low) * rect.height()
        for tick in yticks:
            y = py(tick)
            _line(p, rect.left(), y, rect.right(), y, GRID)
            _text(p, 8, y - 12, 91, 24, _number(tick), 10, MUTED,
                  align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        for value in axis.minor:
            _line(p, px(value), rect.top(), px(value), rect.bottom(), "#2d333d", .8, True)
        for value, text in zip(axis.major, axis.labels):
            x = px(value)
            _line(p, x, rect.top(), x, rect.bottom(), GRID)
            _text(p, x - 70, rect.bottom() + 10, 140, 26, text, 10, MUTED,
                  align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        _text(p, 25, 100, 100, 24, unit, 11, MUTED)
        _text(p, 380, 533, 420, 26, axis.title, 12, MUTED,
              align=Qt.AlignmentFlag.AlignCenter)
        p.save()
        p.setClipRect(rect.adjusted(-1, -1, 1, 1))
        p.setPen(QPen(QColor(accent), 2.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(self._curve(s, values, px, py))
        marker_x = px(s.design_hz)
        _line(p, marker_x, rect.top(), marker_x, rect.bottom(), COPPER, 1.4, True)
        if np.isfinite(design):
            p.setPen(QPen(QColor(EDITOR), 2))
            p.setBrush(QColor(COPPER))
            p.drawEllipse(QPointF(marker_x, py(design)), 5.5, 5.5)
        self._draw_marker(p, rect, values, px, py, unit, 12, 5.5)
        p.restore()
        if not np.any(np.isfinite(values)):
            _text(p, rect.x(), rect.y(), rect.width(), rect.height(),
                  "−∞ dB · 완전 정합" if np.all(np.isneginf(values)) else "표시 가능한 유한한 결과 없음",
                  19, BLUE, align=Qt.AlignmentFlag.AlignCenter)
        if self.stale:
            _text(p, rect.x(), rect.bottom() - 36, rect.width() - 14, 28, "이전 결과 · 입력이 바뀌어 다시 계산 필요",
                  12, AMBER, align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        notes = {
            "impedance": "손실 RLGC 모델의 Re(Zc) · 차동 Z = 2 × Zodd · 점선: 설계 주파수",
            "dcr": "도선 1가닥의 단위 길이 DC 저항 · 주파수에 무관 · 차동 루프는 2가닥 합산",
            "s11": "20 log₁₀|S₁₁| · 0 dB에 가까울수록 반사 큼 · 양단 동일 기준 포트 · 차동쌍은 Sdd11",
            "attenuation": "진행파의 단위 길이 손실 · 도체(표피 효과) 및 유전체 손실 포함",
            "s21": "20 log₁₀|S₂₁| · 0 dB에 가까울수록 손실 작음 · 지정한 길이와 양단 부정합 포함 · 차동쌍은 Sdd21",
        }
        _text(p, 35, 576, 1030, 27, notes[self.metric_key], 11, MUTED)


_owned_application = None


def _application():
    global _owned_application
    if QApplication.instance() is None:
        _owned_application = QApplication([])


def _image(content: Snapshot | Section, metric: str, log_x: bool | None = None,
           marker_hz: float | None = None) -> QImage:
    _application()
    if metric == "section":
        widget = SectionWidget()
        widget.set_section(content.section if isinstance(content, Snapshot) else content)
    else:
        widget = PlotWidget()
        widget.set_metric(metric)
        if log_x is not None:
            widget.set_log_x(log_x)
        widget.set_marker(marker_hz)
        widget.set_snapshot(content)
    image = QImage(1760, 992, QImage.Format.Format_ARGB32)
    p = QPainter(image)
    try:
        widget.draw(p, image.width(), image.height())
    finally:
        p.end()
        widget.deleteLater()
    image.setText("Conductor", content.conductor_name)
    image.setText("ConductivitySPerM", "" if content.conductivity_s_m is None else f"{content.conductivity_s_m:.12g}")
    image.setText("ConductorSourceTemperatureC", "" if content.conductor_temperature_c is None
                  else f"{content.conductor_temperature_c:g}")
    return image


def _atomic_path(path: str | Path):
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix,
                                    dir=destination.parent)
    os.close(handle)
    return destination, Path(name)


def save_png(path: str | Path, content: Snapshot | Section, metric: str, log_x: bool | None = None,
             marker_hz: float | None = None):
    """Write one graph or the section, preserving an existing file on failure.

    metric "section" also accepts a Section, so the drawing can be saved before
    calculating. log_x selects the frequency axis; None uses the metric default.
    marker_hz draws the movable marker with its readout.
    """
    image = _image(content, metric, log_x, marker_hz)
    destination, temporary = _atomic_path(path)
    try:
        if not image.save(str(temporary), "PNG"):
            raise OSError("PNG 파일을 저장하지 못했습니다.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _assumptions(snapshot: Snapshot) -> list[str]:
    generic = [
        "균일한 직선 전송선의 준정적 단면 근사. 단면도는 치수 설명용이며 가로/세로 축척 다름.",
        "마이크로스트립 상부는 공기. 스트립라인은 위·아래 유전체와 두 접지면 적용.",
        "스트립라인의 도체 밖 재료 경계는 도체 두께의 중앙. 마이크로스트립 재료 경계는 도체 하부.",
        "W와 S는 도선 하부 기준. 측벽 각도 90°는 직사각형. 동일층 접지 실드는 바깥 도선에서 G만큼 이격.",
        "차동 Z는 2×Zodd = Vdiff/I. 표시 임피던스는 손실 RLGC 모델의 Re(Zc). DCR은 도선 1가닥 기준.",
        "S11과 S21은 입력·출력의 동일한 기준 임피던스에 연결한 지정 길이 선로의 20 log₁₀|S| (0 dB 이하). 차동쌍은 Sdd11·Sdd21.",
        _conductor_label(snapshot) + ". 온도 보정 없이 지정 전도율 적용.",
        "신호·접지 도체에 동일한 재료와 전도율 적용. 평활 표면·비자성체(μr = 1) 가정.",
        "도체 저항은 DC에서 표피 효과 영역(√f)으로 이어지며 고주파 극한은 증분 인덕턴스 법칙. 이상적인 무한 폭 접지의 DC 저항은 0.",
        "거칠기·비아·커넥터·굽힘·방사·이방성·고차 모드 제외. 실제 설계는 필드 해석/측정으로 확인할 것.",
    ]
    return [*generic, *(WARNINGS[key][1] for key in getattr(snapshot.result, "warnings", ()) if key in WARNINGS)]


def save_pptx(path: str | Path, snapshot: Snapshot, log_x: dict[str, bool] | None = None,
              marker_hz: float | None = None):
    """Export editable conditions/results and the exact Qt figures in a report.

    log_x maps a metric key to its frequency axis; missing keys use the defaults.
    marker_hz adds the movable marker to every graph and to the table of values.
    """
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.oxml.xmlchemy import OxmlElement
    from pptx.util import Inches, Pt

    s, g = snapshot, snapshot.geometry
    log_x = {**LOG_X, **(log_x or {})}
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
        vals = _values(s, key)
        return [_title(s, key), f"{_number(_at_design(s, vals))} {METRICS[key][1]}"]

    mode = "치수 입력" if s.target_ohm is None else f"선폭 역산 · 목표 {s.target_ohm:g} Ω"
    item = slide("01 / 전송선 계산 결과", f"{KINDS[g.kind]}  ·  {mode}  ·  {datetime.now():%Y-%m-%d %H:%M}")
    text(item, .62, 1.52, 11.8, .7, f"설계 주파수 {frequency_text(s.design_hz)}", 32, BLUE, True)
    table(item, ["계산 항목", "설계 주파수 결과"], [metric_row(key) for key in METRICS], y=2.52, h=3.35)
    text(item, .63, 6.2, 11.9, .5,
         f"기준 {s.reference_ohm:g} Ω  ·  선로 길이 {s.length_m * 1000:g} mm  ·  {frequency_text(s.freq_hz[0])} ~ {frequency_text(s.freq_hz[-1])}",
         17, MUTED)

    item = slide("02 / 입력 조건 - 단면 치수", "W·S·G는 도선 하부 기준 · 상부 폭은 측벽 각도로 계산")
    rows = [["구조", KINDS[g.kind]], ["하부 선폭 W / 도체 두께 T", f"{g.width_um:g} / {g.thickness_um:g} µm"],
            ["상부 폭 / 측벽 각도 θ", f"{g.width_um - 2*g.thickness_um/math.tan(math.radians(g.angle_deg)):.4g} µm / {g.angle_deg:g}°"],
            ["하부 유전체 높이 H₁", f"{g.bottom_um:g} µm"],
            ["상부 유전체 높이 H₂", f"{g.top_um:g} µm" if "microstrip" not in g.kind else "해당 없음 · 상부 공기"],
            ["도선 간격 S", f"{g.gap_um:g} µm" if g.kind.startswith("coupled_") else "해당 없음"],
            ["동일층 접지 실드 / 간격 G", f"사용 / {g.clearance_um:g} µm" if g.shield else "사용 안 함"],
            ["길이 / 기준 임피던스", f"{s.length_m * 1000:g} mm / {s.reference_ohm:g} Ω"]]
    table(item, ["항목", "입력값"], rows)

    item = slide("03 / 재료와 주파수 - 계산 데이터", "CMX 주파수 표는 로그 주파수 보간 · 범위 밖 끝값 유지 · 단일 기준점과 직접 입력 재료는 상수 적용")
    material_rows = []
    for side, name, ers, dfs in (("하부", s.bottom_name, s.bottom_er, s.bottom_df),
                                 ("상부", s.top_name, s.top_er, s.top_df)):
        if side == "상부" and "microstrip" in g.kind:
            material_rows.append([side, "공기", "1", "0", "상부 재료 입력 미적용"])
        else:
            material_rows.append([side, name, _number(_at_design(s, ers)), _number(_at_design(s, dfs)),
                                  f"Dk {_number(ers.min())}~{_number(ers.max())}\nDf {_number(dfs.min())}~{_number(dfs.max())}"])
    table(item, ["위치", "재료", "설계 Dk", "설계 Df", "계산 범위 내 변화"], material_rows, y=1.65, h=2.25)
    text(item, .65, 4.2, 11.9, .47, _conductor_label(s), 18)
    text(item, .65, 4.8, 11.9, .4, "신호·접지 동일 재료 · μr = 1 · 온도 보정 없이 지정 전도율 적용", 16, MUTED)
    text(item, .65, 5.45, 11.9, .45,
         f"주파수 범위: {frequency_text(s.freq_hz[0])} ~ {frequency_text(s.freq_hz[-1])} · {len(s.freq_hz):,}개", 18)
    text(item, .65, 6.05, 11.9, .45, f"설계 주파수: {frequency_text(s.design_hz)}", 18)

    with tempfile.TemporaryDirectory(prefix="zcalc-report-") as folder:
        for number, metric in enumerate(["section", *METRICS], start=4):
            label = "단면 구조 - 연결과 치수" if metric == "section" else _title(s, metric)
            item = slide(f"{number:02d} / {label}")
            picture = Path(folder) / f"{metric}.png"
            save_png(picture, s, metric, log_x.get(metric), marker_hz)
            item.shapes.add_picture(str(picture), Inches(1.55), Inches(1.1), height=Inches(5.78))

        item = slide("10 / 수치 결과 - 대표 주파수", "원본 계산점 중 대표값 · 설계 주파수는 별도 보간값")
        indices = sorted(set(np.linspace(0, len(s.freq_hz) - 1, 5).astype(int)))
        rows = []
        for index in indices:
            rows.append([frequency_text(s.freq_hz[index]), *[_number(_values(s, key)[index]) for key in METRICS]])
        rows.append(["설계 " + frequency_text(s.design_hz), *[_number(_at_design(s, _values(s, key))) for key in METRICS]])
        if marker_hz is not None and s.freq_hz[0] <= marker_hz <= s.freq_hz[-1]:
            rows.append(["마커 " + frequency_text(marker_hz, 4),
                         *[_number(_at(s, _values(s, key), marker_hz)) for key in METRICS]])
        pair = g.kind.startswith("coupled_")
        table(item, ["주파수", "Z (Ω)", "DCR (Ω/m)", "Sdd11 (dB)" if pair else "S11 (dB)", "전파 손실 (dB/m)",
                     "Sdd21 (dB)" if pair else "S21 (dB)"], rows, h=4.25)
        text(item, .64, 6.15, 12, .45, "DCR: 도선 1가닥 기준. 차동 Z = 2 × Zodd = Vdiff/I.", 15, MUTED)

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
