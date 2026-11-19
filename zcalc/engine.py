"""Quasi-TEM cross-section solver, SI outputs and micrometre geometry.

A conforming, graded triangular P1 finite-element solution of div(eps grad V)=0
provides C and dielectric energy participation. A second vacuum solve gives
L_external = 1/(c**2 C_vacuum). The half-domain uses electric symmetry for
an odd coupled mode, and magnetic symmetry for a single trace. Coupled results
are DIFFERENTIAL (V_right - V_left)/I_right, not the odd-mode impedance. A
second solve with magnetic symmetry gives the even mode of a pair, per line;
both modes together give the four single-ended ports of the pair.
Microstrip is exposed to air: top_um/top_er/top_df are ignored. Stripline's
bottom_um and top_um are copper-face-to-plane distances. The two stripline
dielectrics meet at copper mid-thickness outside the trace. Copper's bottom
width and bottom edge gap are inputs; sidewall angle is measured from the
horizontal. Optional infinite grounded coplanar rails begin clearance_um from
the outside bottom edges, with the same thickness and rectangular sidewalls.

Loss: one smooth nonmagnetic conductor material for traces, planes and rails.
The default is copper, rho=1.724e-8 ohm m at 20 C; conductivity_s_m selects
another finite, positive conductivity in S/m. DCR is for ONE signal trace.
The skin-effect limit follows Wheeler's incremental-inductance rule,
R = (Rs/mu0) dL/dn with every metal surface (trace, planes, rails) receding,
evaluated as the shape derivative of the vacuum capacitance. A slab coth
internal-impedance transition joins signal DCR to that skin limit; return
planes have zero DC resistance (infinite width). Dielectric loss uses
first-order electric-energy participation. No roughness, radiation,
connectors, vias, plating, anisotropy or higher-order dispersion. Open
boundaries are truncated at 20 characteristic dimensions with natural Neumann
conditions. This is an engineering estimate, not a fabrication signoff field
solver. Sharp corners are assumed; real copper rounding and roughness limit
conductor-loss accuracy, as does the slab transition where the skin depth is
comparable to the copper thickness.

Method references: C. R. Paul, Analysis of Multiconductor Transmission Lines,
2nd ed., Wiley-IEEE, 2008, chs. 4-5 (p.u.l. parameters and numerical methods);
H. A. Wheeler, "Formulas for the skin effect", Proc. IRE 30(9), 1942,
https://doi.org/10.1109/JRPROC.1942.232015 (incremental-inductance rule).
Benchmarks: E. Hammerstad & O. Jensen, IEEE MTT-S 1980, pp. 407-409,
https://doi.org/10.1109/MWSYM.1980.1124303 (microstrip); S. B. Cohn, IRE Trans.
MTT 2(2), 1954, https://doi.org/10.1109/TMTT.1954.1124875 and 3(5), 1955,
https://doi.org/10.1109/TMTT.1955.1124973 (exact zero-thickness single and
edge-coupled stripline); D. M. Pozar, Microwave Engineering, 4th ed., Wiley,
2012, sec. 3.7 (stripline conductor attenuation, wide-strip branch).
Copper reference: NBS Circular 31, 4th ed.,
https://nvlpubs.nist.gov/nistpubs/Legacy/circ/nbscircular31e4.pdf .
Analytic homogeneous TEM scaling and parallel-plate limits are test cases.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C0 = 1.0 / math.sqrt(EPS0 * MU0)
COPPER_RHO = 1.724e-8
COPPER_CONDUCTIVITY_S_M = 1 / COPPER_RHO
KINDS = ("microstrip", "stripline", "coupled_microstrip", "coupled_stripline")
# Result.warnings keys: conditions under which the quasi-static estimate degrades.
WARNINGS = {
    "wavelength": "Cross-section exceeds one tenth wavelength; higher-order/full-wave validation required.",
    "loss_tangent": "Dielectric loss uses a small-loss perturbation; tan(delta) > 0.1 reduces accuracy.",
    "aspect_ratio": "Dimension ratio above 100:1; mesh accuracy needs independent convergence validation.",
}


@dataclass(frozen=True)
class Geometry:
    kind: str = "microstrip"
    width_um: float = 200.0
    thickness_um: float = 35.0
    bottom_um: float = 100.0
    top_um: float = 100.0
    gap_um: float = 100.0
    angle_deg: float = 90.0
    shield: bool = False
    clearance_um: float = 200.0

    @property
    def coupled(self) -> bool:
        return self.kind.startswith("coupled_")

    @property
    def stripline(self) -> bool:
        return self.kind.endswith("stripline") and not self.kind.endswith("microstrip")

    @property
    def top_width_um(self) -> float:
        return self.width_um - 2 * self.thickness_um / math.tan(math.radians(self.angle_deg))

    def active_dimensions(self) -> dict[str, float]:
        """Dimensions the selected structure uses, keyed by their on-screen names."""
        active = {"하부 선폭 W": self.width_um, "도선 두께 T": self.thickness_um,
                  "하부 유전체 H₁": self.bottom_um}
        if self.stripline:
            active["상부 유전체 H₂"] = self.top_um
        if self.coupled:
            active["도선 간격 S"] = self.gap_um
        if self.shield:
            active["실드 간격 G"] = self.clearance_um
        return active

    def validate(self) -> None:
        """Raise ValueError with a user-facing (Korean) message for unusable input."""
        if self.kind not in KINDS:
            raise ValueError(f"지원하지 않는 전송선 구조입니다: {self.kind}")
        active = self.active_dimensions()
        for name, value in active.items():
            if not np.isfinite(value) or not 0.001 <= value <= 1e7:
                raise ValueError(f"{name}: 0.001–10,000,000 µm 범위의 값을 입력하세요.")
        if not np.isfinite(self.angle_deg) or not 45 <= self.angle_deg <= 90:
            raise ValueError("측벽 각도: 45°–90° 범위의 값을 입력하세요.")
        if self.top_width_um <= max(1e-6, self.width_um * 1e-5):
            raise ValueError("측벽 각도와 도선 두께에 비해 하부 선폭이 좁아 도선 상부 폭이 남지 않습니다.")
        if max(active.values()) / min(active.values()) > 1e5:
            raise ValueError("단면 치수의 최대/최소 비가 격자 한계(100,000:1)를 넘습니다.")


@dataclass
class Result:
    frequency_hz: np.ndarray
    impedance_ohm: np.ndarray
    s11_db: np.ndarray
    s21_db: np.ndarray
    attenuation_db_m: np.ndarray
    dcr_ohm_m: float
    quasistatic_impedance_ohm: np.ndarray
    complex_impedance_ohm: np.ndarray
    resistance_ohm_m: np.ndarray
    inductance_h_m: np.ndarray
    conductance_s_m: np.ndarray
    capacitance_f_m: np.ndarray
    effective_er: np.ndarray
    s11: np.ndarray
    s21: np.ndarray
    propagation_constant_m: np.ndarray
    warnings: tuple[str, ...]
    # Single-ended ports, shape (frequencies, ports, ports); None without ports=True.
    # Single line: port 1 near end, port 2 far end, each referenced to reference_ohm.
    # Pair: ports 1/2 are the near/far end of one line and 3/4 of the other
    # (through paths 1-2 and 3-4), each referenced to port_ohm = reference_ohm / 2.
    s_parameters: np.ndarray | None = None
    port_ohm: float | None = None
    even_impedance_ohm: np.ndarray | None = None  # complex, per line; pairs only
    even_propagation_constant_m: np.ndarray | None = None


def _cosine_segment(a: float, b: float, count: int) -> np.ndarray:
    return a + (b - a) * (1 - np.cos(np.linspace(0, np.pi, count + 1))) / 2


def _graded_segment(a: float, b: float, first_step: float) -> np.ndarray:
    half = _outward(a, (a + b) / 2, first_step)
    return np.concatenate((half, a + b - half[-2::-1]))


def _outward(a: float, b: float, first_step: float) -> np.ndarray:
    values = [a]
    step = first_step
    while values[-1] + step < b:
        values.append(values[-1] + step)
        step *= 1.19
    values.append(b)
    return np.asarray(values)


class _Section:
    def __init__(self, g: Geometry, even: bool = False):
        # Coordinates are micrometres: electrostatic 2D energy is scale invariant.
        w, t, h = g.width_um, g.thickness_um, g.bottom_um
        inner = g.gap_um / 2 if g.coupled else 0.0
        outer = inner + w if g.coupled else w / 2
        scale = max(h, g.top_um if g.stripline else h, outer, t)
        rail = outer + g.clearance_um
        xmax = (rail if g.shield else outer) + 20 * scale
        small = min(h, w, g.gap_um if g.coupled else w,
                    g.clearance_um if g.shield else w)
        # Logical columns: [symmetry plane, (inner edge), outer edge, (rail edge), open boundary].
        xparts, anchors = [], [0.0]
        if g.coupled:
            xparts.append(_graded_segment(0, inner, small / 120)[:-1])
            anchors.append(inner)
        first = sum(map(len, xparts))
        xparts.append(_cosine_segment(inner, outer, 30)[:-1])
        anchors.append(outer)
        last = sum(map(len, xparts))
        if g.shield:
            xparts.append(_graded_segment(outer, rail, small / 120)[:-1])
            anchors.append(rail)
            rail_column = sum(map(len, xparts))
            xparts.append(_outward(rail, xmax, small / 24))
        else:
            xparts.append(_outward(outer, xmax, small / 120))
        anchors.append(xmax)
        x = np.concatenate(xparts)
        yparts = [_cosine_segment(-h, 0, 36)[:-1], np.linspace(0, t, 7)[:-1]]
        bottom_row, top_row = len(yparts[0]), len(yparts[0]) + len(yparts[1])
        if g.stripline:
            yparts.append(_cosine_segment(t, t + g.top_um, 36))
        else:
            yparts.append(_outward(t, t + 20 * scale, small / 35))
        y = np.concatenate(yparts)
        nx, ny = len(x), len(y)
        self.g, self.x, self.y = g, x, y
        self.inner, self.outer, self.rail, self.xmax = inner, outer, rail, xmax
        anchors = np.asarray(anchors)
        self.segment = np.clip(np.searchsorted(anchors, x, side="right") - 1, 0, len(anchors) - 2)
        self.weight = (x - anchors[self.segment]) / (anchors[self.segment + 1] - anchors[self.segment])
        base = (np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)).ravel()
        self.tri = np.concatenate((np.column_stack((base, base + 1, base + nx)),
                                   np.column_stack((base + 1, base + nx + 1, base + nx))))
        self.xy = self._nodes(0.0)
        self.area, self.grad = self._elements(self.xy)
        self.local = self.area[:, None, None] * np.einsum("tix,tjx->tij", self.grad, self.grad)
        # Conductors are sets of logical grid nodes, independent of coordinate rounding.
        column, row = np.tile(np.arange(nx), ny), np.repeat(np.arange(ny), nx)
        copper_rows = (row >= bottom_row) & (row <= top_row)
        signal = copper_rows & (column >= first) & (column <= last)
        ground = row == 0
        if g.stripline:
            ground |= row == ny - 1
        if g.coupled and not even:
            ground |= column == 0  # odd-mode symmetry plane: zero potential, not metal
        if g.shield:
            ground |= copper_rows & (column >= rail_column)
        # The stripline dielectrics meet at copper mid-thickness, a grid row.
        interface = (bottom_row + top_row) // 2 if g.stripline else bottom_row
        self.lower = np.tile(np.repeat(np.arange(ny - 1), nx - 1), 2) < interface
        self.boundary = signal.astype(float)
        self.unknown = np.flatnonzero(~(signal | ground))
        self.rows = np.repeat(self.tri, 3, axis=1).ravel()
        self.cols = np.tile(self.tri, (1, 3)).ravel()
        # The half-domain energy is half a single line, one line of an even
        # mode, or twice the differential capacitance of an odd mode.
        self.factor = (1.0 if even else 0.5) if g.coupled else 2.0
        bottom, top, potential = self.solve(1.0, 1.0)
        # Wheeler's incremental-inductance rule: R_hf = (Rs/mu0) dL/dn with every
        # metal surface receding by n. With L = 1/(c^2 C) this is the shape derivative
        # of the vacuum capacitance. For the discrete minimum-energy potential the
        # derivative equals the energy change at frozen potential (envelope theorem),
        # so it converges like the capacitance itself. A direct surface integral of
        # E^2 converges only as h^(1/3) at 90 degree corners (7-16% low on this grid).
        # The step stays small against every dimension, including gaps far thinner than the copper.
        step = 1e-3 * min(g.top_width_um, *g.active_dimensions().values())
        receded, grown = (self._stored(self._nodes(d), potential).sum() for d in (step, -step))
        stored = (bottom + top) / (EPS0 * self.factor)
        surface_factor = 1e6 * (grown - receded) / (2 * step * self.factor * stored ** 2)
        self.vacuum = bottom, top, surface_factor

    def _nodes(self, recess: float) -> np.ndarray:
        """Node coordinates with every metal surface moved recess um into the metal.

        The logical grid is kept, making capacitance a smooth function of recess.
        """
        g = self.g
        t = g.thickness_um - 2 * recess
        y = np.interp(self.y, [self.y[0], 0, g.thickness_um, self.y[-1]],
                      [self.y[0] - recess, recess, recess + t,
                       self.y[-1] + (recess if g.stripline else 0)])
        angle = math.radians(g.angle_deg)
        # A sidewall moved along its normal shifts the bottom corner by recess (1/sin + 1/tan).
        edge = (recess * (1 / math.sin(angle) + 1 / math.tan(angle))
                + t / math.tan(angle) * np.clip((y - recess) / t, 0, 1))
        columns = [np.zeros_like(y)]
        if g.coupled:
            columns.append(self.inner + edge)
        columns.append(self.outer - edge)
        if g.shield:
            columns.append(np.full_like(y, self.rail + recess))
        columns.append(np.full_like(y, self.xmax))
        dest = np.column_stack(columns)
        xx = dest[:, self.segment] * (1 - self.weight) + dest[:, self.segment + 1] * self.weight
        return np.column_stack((xx.ravel(), np.repeat(y, len(self.x))))

    def _elements(self, xy: np.ndarray):
        pts = xy[self.tri]
        determinant = ((pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
                       - (pts[:, 2, 0] - pts[:, 0, 0]) * (pts[:, 1, 1] - pts[:, 0, 1]))
        grad = np.stack((pts[:, [1, 2, 0], 1] - pts[:, [2, 0, 1], 1],
                         pts[:, [2, 0, 1], 0] - pts[:, [1, 2, 0], 0]), axis=2) / determinant[:, None, None]
        return np.abs(determinant) / 2, grad

    def _stored(self, xy: np.ndarray, potential: np.ndarray) -> np.ndarray:
        """Integral of |grad V|^2 over each element of the grid at xy."""
        area, grad = (self.area, self.grad) if xy is self.xy else self._elements(xy)
        field = np.einsum("ti,tix->tx", potential[self.tri], grad)
        return area * np.einsum("tx,tx->t", field, field)

    def solve(self, bottom_er: float, top_er: float):
        eps = np.where(self.lower, bottom_er, top_er)
        values = (self.local * eps[:, None, None]).ravel()
        matrix = coo_matrix((values, (self.rows, self.cols)),
                            shape=(len(self.xy), len(self.xy))).tocsr()
        rhs = -(matrix @ self.boundary)[self.unknown]
        potential = self.boundary.copy()
        potential[self.unknown] = spsolve(matrix[self.unknown][:, self.unknown], rhs)
        energy = self._stored(self.xy, potential) * eps * EPS0 * self.factor
        return energy[self.lower].sum(), energy[~self.lower].sum(), potential


@lru_cache(maxsize=16)
def _section(g: Geometry, even: bool = False) -> _Section:
    return _Section(g, even)


def _array(value, frequency: np.ndarray, label: str, low: float, high: float) -> np.ndarray:
    try:
        a = np.broadcast_to(np.asarray(value, dtype=float), frequency.shape)
    except ValueError as error:
        raise ValueError(f"{label}: 값 하나 또는 주파수와 같은 개수의 값이 필요합니다.") from error
    if not np.all(np.isfinite(a)) or np.any(a < low) or np.any(a > high):
        raise ValueError(f"{label}: {low:g}–{high:g} 범위의 값을 입력하세요.")
    return a


def _capacitances(section: _Section, eb: np.ndarray, et: np.ndarray):
    ratios = eb / et
    distinct = np.unique(ratios)
    count = max(9, int(np.ceil(np.log(distinct[-1] / distinct[0]) / 0.25)) + 1)
    samples = distinct if len(distinct) <= count else np.geomspace(distinct[0], distinct[-1], count)
    energies = []
    for ratio in samples:
        if abs(ratio - 1) < 1e-13:
            energies.append(section.vacuum[:2])
        else:
            energies.append(section.solve(float(ratio), 1.0)[:2])
    energies = np.asarray(energies)
    if len(samples) == 1:
        bottom = np.full_like(eb, energies[0, 0])
        top = np.full_like(eb, energies[0, 1])
    elif len(distinct) <= count:
        ids = np.searchsorted(samples, ratios)
        bottom, top = energies[ids].T
    else:
        bottom, top = PchipInterpolator(np.log(samples), energies, axis=0)(np.log(ratios)).T
    return bottom * et, top * et


@dataclass
class _Line:
    """Per-unit-length parameters of one mode."""
    resistance: np.ndarray
    inductance: np.ndarray
    conductance: np.ndarray
    capacitance: np.ndarray
    vacuum_capacitance: float
    impedance: np.ndarray  # complex characteristic impedance
    gamma: np.ndarray  # complex propagation constant, 1/m

    @property
    def external_inductance(self) -> float:
        return 1 / (C0 ** 2 * self.vacuum_capacitance)


def _resistivity(conductivity_s_m) -> float:
    """Validate one finite conductor conductivity and its representable reciprocal."""
    message = "도체 전도율: 0보다 큰 유한한 S/m 값을 입력하세요."
    try:
        if np.ndim(conductivity_s_m) != 0:
            raise ValueError(message)
        conductivity = float(conductivity_s_m)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(conductivity) or conductivity <= 0:
        raise ValueError(message)
    rho = 1 / conductivity
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("도체 전도율이 수치 계산 범위를 벗어납니다.")
    return rho


def _line(section: _Section, f, eb, db, et, dt, rdc: float, rho: float) -> _Line:
    """RLGC of the mode a section was solved for; rdc is the DC resistance of its current path."""
    cb, ct = _capacitances(section, eb, et)
    capacitance = cb + ct
    cvac = sum(section.vacuum[:2])
    omega = 2 * np.pi * f
    conductance = omega * (cb * db + ct * dt)
    # Skin effect: the resistance rises from rdc to Rs(f) * surface_factor, Rs = sqrt(pi f mu0 rho),
    # and the same internal impedance adds inductance. A slab of equivalent thickness joins the
    # two limits; where the skin depth is near the conductor thickness this crossover is an
    # estimate, not a diffusion solve.
    if not np.isfinite(rdc) or rdc <= 0:
        raise ValueError("도체 전도율에 따른 저항이 수치 계산 범위를 벗어납니다.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        depth = section.vacuum[2] * (rho / rdc)
        # Separate square roots keep frequency / rho from overflowing for high conductivity.
        z = (1 + 1j) * (np.sqrt(omega * MU0 / 2) / math.sqrt(rho) * depth)
        internal = rdc * z
        transition = (z.real >= 1e-4) & (z.real < 20)
        internal[transition] /= np.tanh(z[transition])
        small = z.real < 1e-4
        internal[small] = rdc * (1 + z[small] ** 2 / 3)
        inductance = 1 / (C0 ** 2 * cvac) + internal.imag / omega
        series = internal.real + 1j * omega * inductance
        shunt = conductance + 1j * omega * capacitance
        impedance = np.sqrt(series) / np.sqrt(shunt)
        gamma = np.sqrt(series) * np.sqrt(shunt)
    if not all(np.all(np.isfinite(value)) for value in (internal, inductance, impedance, gamma)):
        raise ValueError("도체 전도율에 따른 선로 특성이 수치 계산 범위를 벗어납니다.")
    return _Line(internal.real, inductance, conductance, capacitance, cvac,
                 impedance, gamma)


def _two_port(impedance, gamma, length_m: float, reference_ohm: float):
    """S11 and S21 of a uniform line between two equal real reference impedances."""
    if length_m == 0:
        return np.zeros_like(impedance), np.ones_like(impedance)
    # Decaying-wave formulation avoids cosh/sinh overflow on long/lossy lines.
    rho = (impedance - reference_ohm) / (impedance + reference_ohm)
    travel = np.exp(-gamma * length_m)
    denominator = 1 - rho ** 2 * travel ** 2
    return rho * (1 - travel ** 2) / denominator, (1 - rho ** 2) * travel / denominator


def analyze(geometry: Geometry, freq_hz, bottom_er=4.2, bottom_df=0.02,
            top_er=4.2, top_df=0.02, length_m=0.1, reference_ohm=50.0, ports=True,
            *, conductivity_s_m=COPPER_CONDUCTIVITY_S_M) -> Result:
    """Analyze a single or odd differential line; both ports use reference_ohm.

    impedance_ohm is Re(Zc) of lossy RLGC. quasistatic_impedance_ohm is
    sqrt(L_external/C), the geometric impedance often used for PCB synthesis.
    S11/S21 include propagation, finite length and mismatch at both ends;
    s11_db and s21_db are 20 log10(abs(S)), so they are zero or negative, and a
    matched S11 is floored at -300 dB for plotting. For a pair these are the
    differential Sdd11/Sdd21 against the differential reference_ohm.
    With ports=True, s_parameters holds the single-ended ports (see Result);
    a pair then also needs its even mode. Frequency must be strictly increasing
    and >0 Hz. Result.warnings holds the keys of WARNINGS that apply. ValueError
    messages are user-facing Korean text.
    conductivity_s_m applies to all traces, planes and rails, with relative
    permeability fixed at one; no plating, roughness or temperature correction.
    Infinity is not interpreted as a perfect electric conductor.
    """
    geometry.validate()
    f = np.asarray(freq_hz, dtype=float)
    if (f.ndim != 1 or not len(f) or not np.all(np.isfinite(f))
            or np.any(f < 1e-3) or np.any(f > 1e15) or np.any(np.diff(f) <= 0)):
        raise ValueError("주파수는 0.001 Hz–1 PHz 범위에서 오름차순이어야 합니다.")
    if not np.isfinite(length_m) or length_m < 0:
        raise ValueError("선로 길이: 0 이상의 값을 입력하세요.")
    if not np.isfinite(reference_ohm) or reference_ohm <= 0:
        raise ValueError("양단 기준 임피던스: 0보다 큰 값을 입력하세요.")
    rho = _resistivity(conductivity_s_m)
    eb = _array(bottom_er, f, "하부 유전체 Dk", 1, 100)
    db = _array(bottom_df, f, "하부 유전체 Df", 0, 1)
    et = _array(top_er, f, "상부 유전체 Dk", 1, 100) if geometry.stripline else np.ones_like(f)
    dt = _array(top_df, f, "상부 유전체 Df", 0, 1) if geometry.stripline else np.zeros_like(f)
    area_m2 = (geometry.width_um + geometry.top_width_um) / 2 * geometry.thickness_um * 1e-12
    dcr = rho / area_m2
    if not np.isfinite(dcr) or dcr <= 0:
        raise ValueError("도체 전도율에 따른 저항이 수치 계산 범위를 벗어납니다.")
    # The differential current returns through the other trace: two conductors in series.
    line = _line(_section(geometry), f, eb, db, et, dt, dcr * (2 if geometry.coupled else 1), rho)
    s11, s21 = _two_port(line.impedance, line.gamma, length_m, reference_ohm)
    s_parameters = port_ohm = even = None
    if ports and geometry.coupled:
        port_ohm = reference_ohm / 2
        even = _line(_section(geometry, True), f, eb, db, et, dt, dcr, rho)
        e11, e21 = _two_port(even.impedance, even.gamma, length_m, port_ohm)
        # Each line of the odd mode against port_ohm is the differential line against reference_ohm.
        reflect, near = (e11 + s11) / 2, (e11 - s11) / 2
        through, far = (e21 + s21) / 2, (e21 - s21) / 2
        rows = [[reflect, through, near, far], [through, reflect, far, near],
                [near, far, reflect, through], [far, near, through, reflect]]
    elif ports:
        port_ohm = reference_ohm
        rows = [[s11, s21], [s21, s11]]
    if ports:
        s_parameters = np.moveaxis(np.array(rows), -1, 0)
    warnings = []
    dimension_m = (geometry.bottom_um + geometry.thickness_um +
                   (geometry.top_um if geometry.stripline else geometry.width_um)) * 1e-6
    if f[-1] * dimension_m * np.sqrt(max(eb.max(), et.max())) / C0 > 0.1:
        warnings.append("wavelength")
    if max(db.max(), dt.max()) > 0.1:
        warnings.append("loss_tangent")
    active = geometry.active_dimensions().values()
    if max(active) / min(active) > 100:
        warnings.append("aspect_ratio")
    return Result(f.copy(), line.impedance.real, 20 * np.log10(np.maximum(abs(s11), 1e-15)),
                  20 * np.log10(np.maximum(abs(s21), 1e-300)), 20 / np.log(10) * line.gamma.real, dcr,
                  np.sqrt(line.external_inductance / line.capacitance), line.impedance,
                  line.resistance, line.inductance, line.conductance, line.capacitance,
                  line.capacitance / line.vacuum_capacitance, s11, s21, line.gamma, tuple(warnings),
                  s_parameters, port_ohm, None if even is None else even.impedance,
                  None if even is None else even.gamma)


def synthesize_width(geometry: Geometry, target_ohm: float, frequency_hz: float,
                     bottom_er=4.2, bottom_df=0.02, top_er=4.2, top_df=0.02,
                     *, conductivity_s_m=COPPER_CONDUCTIVITY_S_M) -> float:
    """Return bottom width in um matching Re(lossy Zc) at one design frequency.

    The gap, thickness, height, angle and shield clearance remain fixed. The
    result is reanalyzed and must agree within 0.2% (or 0.05 ohm).
    """
    # A seed width need not be manufacturable: width is the quantity being solved.
    validation_width = max(4 * geometry.thickness_um, geometry.bottom_um, 1.0)
    replace(geometry, width_um=validation_width).validate()
    if not np.isfinite(target_ohm) or target_ohm <= 0:
        raise ValueError("목표 임피던스: 0보다 큰 값을 입력하세요.")
    _resistivity(conductivity_s_m)
    inset = 2 * geometry.thickness_um / math.tan(math.radians(geometry.angle_deg))
    height = min(geometry.bottom_um, geometry.top_um) if geometry.stripline else geometry.bottom_um
    lo = max(inset * 1.001, height * 0.01, 0.01)
    hi = max(height * 30, inset * 2)

    def error(width):
        result = analyze(replace(geometry, width_um=float(width)), [frequency_hz],
                         bottom_er, bottom_df, top_er, top_df, length_m=0, ports=False,
                         conductivity_s_m=conductivity_s_m)
        return float(result.impedance_ohm[0] - target_ohm)

    elo, ehi = error(lo), error(hi)
    if elo * ehi > 0:
        raise ValueError(f"목표 임피던스가 선폭 탐색 범위 밖입니다. 선폭 {lo:.3g}–{hi:.3g} µm에서 "
                         f"{ehi + target_ohm:.4g}–{elo + target_ohm:.4g} Ω만 얻을 수 있습니다.")
    width = brentq(error, lo, hi, xtol=max(1e-4, height * 1e-5), rtol=1e-5)
    if abs(error(width)) > max(0.05, target_ohm * 0.002):
        raise ValueError("선폭 역산 결과가 목표 임피던스 허용차를 만족하지 못했습니다.")
    return float(width)
