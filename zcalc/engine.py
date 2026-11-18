"""Quasi-TEM cross-section solver, SI outputs and micrometre geometry.

A conforming, graded triangular P1 finite-element solution of div(eps grad V)=0
provides C and dielectric energy participation. A second vacuum solve gives
L_external = 1/(c**2 C_vacuum). The half-domain uses electric symmetry for
an odd coupled mode, and magnetic symmetry for a single trace. Coupled results
are DIFFERENTIAL (V_right - V_left)/I_right, not the odd-mode impedance.
Microstrip is exposed to air: top_um/top_er/top_df are ignored. Stripline's
bottom_um and top_um are copper-face-to-plane distances. The two stripline
dielectrics meet at copper mid-thickness outside the trace. Copper's bottom
width and bottom edge gap are inputs; sidewall angle is measured from the
horizontal. Optional infinite grounded coplanar rails begin clearance_um from
the outside bottom edges, with the same thickness and rectangular sidewalls.

Loss: smooth copper, rho=1.724e-8 ohm m at 20 C. DCR is for ONE signal trace.
Vacuum surface fields estimate skin/proximity loss including ideal connected
planes/rails. A slab coth internal-impedance transition joins signal DCR to
that skin limit; return planes have zero DC resistance (infinite width).
Dielectric loss uses first-order electric-energy participation. No roughness,
radiation, connectors, vias, plating, anisotropy or higher-order dispersion.
Finite grounded planes in stripline; open microstrip boundaries are truncated
at 20 characteristic dimensions with natural Neumann boundaries. This is an
engineering estimate, not a fabrication signoff field solver. Corner singularity
and unknown copper rounding especially limit conductor-loss accuracy.

Method references: C. R. Paul, Analysis of Multiconductor Transmission Lines,
2nd ed., Wiley-IEEE, 2008, chs. 4-5 (p.u.l. parameters and numerical methods);
E. Hammerstad & O. Jensen, IEEE MTT-S 1980, pp. 407-409,
https://doi.org/10.1109/MWSYM.1980.1124303 (microstrip benchmark).
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
KINDS = ("microstrip", "stripline", "coupled_microstrip", "coupled_stripline")


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

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown cross-section kind: {self.kind}")
        required = [self.width_um, self.thickness_um, self.bottom_um]
        if self.stripline:
            required.append(self.top_um)
        if self.coupled:
            required.append(self.gap_um)
        if self.shield:
            required.append(self.clearance_um)
        if not all(np.isfinite(v) and 0.001 <= v <= 1e7 for v in required):
            raise ValueError("Active geometry dimensions must be between 0.001 and 10000000 um.")
        if not np.isfinite(self.angle_deg) or not 45 <= self.angle_deg <= 90:
            raise ValueError("Sidewall angle must be between 45 and 90 degrees.")
        if self.top_width_um <= max(1e-6, self.width_um * 1e-5):
            raise ValueError("Sidewalls leave no positive copper top width.")
        if max(required) / min(required) > 1e5:
            raise ValueError("Geometry aspect ratio exceeds the mesh limit (100000:1).")


@dataclass
class Result:
    frequency_hz: np.ndarray
    impedance_ohm: np.ndarray
    return_loss_db: np.ndarray
    insertion_loss_db: np.ndarray
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
    def __init__(self, g: Geometry):
        # Coordinates are micrometres: electrostatic 2D energy is scale invariant.
        w, t, h = g.width_um, g.thickness_um, g.bottom_um
        inner = g.gap_um / 2 if g.coupled else 0.0
        outer = inner + w if g.coupled else w / 2
        scale = max(h, g.top_um if g.stripline else h, outer, t)
        rail = outer + g.clearance_um if g.shield else np.inf
        xmax = (rail if g.shield else outer) + 20 * scale
        small = min(h, w, g.gap_um if g.coupled else w,
                    g.clearance_um if g.shield else w)
        xparts = []
        if g.coupled:
            xparts.append(_graded_segment(0, inner, small / 120)[:-1])
        xparts.append(_cosine_segment(inner, outer, 30)[:-1])
        anchors = [0, inner, outer] if g.coupled else [0, outer]
        if g.shield:
            xparts.append(_graded_segment(outer, rail, small / 120)[:-1])
            anchors.append(rail)
            xparts.append(_outward(rail, xmax, small / 24))
        else:
            xparts.append(_outward(outer, xmax, small / 120))
        x = np.concatenate(xparts)
        yparts = [_cosine_segment(-h, 0, 36)[:-1], np.linspace(0, t, 7)[:-1]]
        if g.stripline:
            yparts.append(_cosine_segment(t, t + g.top_um, 36))
        else:
            yparts.append(_outward(t, t + 20 * scale, small / 35))
        y = np.concatenate(yparts)
        xx = np.tile(x, (len(y), 1))
        inset = t / math.tan(math.radians(g.angle_deg))
        for j, yy in enumerate(y):
            fraction = np.clip(yy / t, 0, 1)
            dest = list(anchors)
            if g.coupled:
                dest[1] += inset * fraction
            dest[2 if g.coupled else 1] -= inset * fraction
            xx[j] = np.interp(x, anchors + [xmax], dest + [xmax])
        self.xy = np.column_stack((xx.ravel(), np.repeat(y, len(x))))
        nx, ny = len(x), len(y)
        base = (np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)).ravel()
        triangles = np.concatenate((np.column_stack((base, base + 1, base + nx)),
                                    np.column_stack((base + 1, base + nx + 1, base + nx))))
        self.tri = triangles
        pts = self.xy[triangles]
        determinant = ((pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
                       - (pts[:, 2, 0] - pts[:, 0, 0]) * (pts[:, 1, 1] - pts[:, 0, 1]))
        self.area = np.abs(determinant) / 2
        self.grad = np.stack((pts[:, [1, 2, 0], 1] - pts[:, [2, 0, 1], 1],
                              pts[:, [2, 0, 1], 0] - pts[:, [1, 2, 0], 0]), axis=2) / determinant[:, None, None]
        self.local = self.area[:, None, None] * np.einsum("tix,tjx->tij", self.grad, self.grad)
        interface = t / 2 if g.stripline else 0
        self.lower = pts[:, :, 1].mean(axis=1) < interface
        xp, yp = self.xy.T
        inset_y = inset * np.clip(yp / t, 0, 1)
        signal = ((yp >= -1e-10) & (yp <= t + 1e-10)
                  & (xp >= (inner + inset_y if g.coupled else 0) - 1e-9)
                  & (xp <= outer - inset_y + 1e-9))
        ground = np.isclose(yp, -h, rtol=0, atol=1e-9)
        if g.stripline:
            ground |= np.isclose(yp, t + g.top_um, rtol=0, atol=1e-9)
        if g.coupled:
            ground |= np.isclose(xp, 0, rtol=0, atol=1e-9)
        rail_nodes = ((xp >= rail - 1e-9) & (yp >= -1e-10) & (yp <= t + 1e-10))
        ground |= rail_nodes
        self.fixed = signal | ground
        self.boundary = signal.astype(float)
        self.unknown = np.flatnonzero(~self.fixed)
        self.rows = np.repeat(triangles, 3, axis=1).ravel()
        self.cols = np.tile(triangles, (1, 3)).ravel()
        self.factor = 0.5 if g.coupled else 2.0
        self.g = g
        # Exterior conductor edges only; the odd-mode symmetry plane is not metal.
        metal = signal | rail_nodes | np.isclose(yp, -h, rtol=0, atol=1e-9)
        if g.stripline:
            metal |= np.isclose(yp, t + g.top_um, rtol=0, atol=1e-9)
        edge_tri, edge_length = [], []
        allmetal = metal[triangles].all(axis=1)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            on_edge = metal[triangles[:, a]] & metal[triangles[:, b]] & ~allmetal
            ids = np.flatnonzero(on_edge)
            edge_tri.extend(ids)
            edge_length.extend(np.linalg.norm(pts[ids, a] - pts[ids, b], axis=1))
        self.surface_tri = np.asarray(edge_tri, dtype=int)
        self.surface_length_m = np.asarray(edge_length) * 1e-6
        self.vacuum = self.solve(1.0, 1.0, surface=True)

    def solve(self, bottom_er: float, top_er: float, surface: bool = False):
        eps = np.where(self.lower, bottom_er, top_er)
        values = (self.local * eps[:, None, None]).ravel()
        matrix = coo_matrix((values, (self.rows, self.cols)),
                            shape=(len(self.xy), len(self.xy))).tocsr()
        rhs = -(matrix @ self.boundary)[self.unknown]
        potential = self.boundary.copy()
        potential[self.unknown] = spsolve(matrix[self.unknown][:, self.unknown], rhs)
        field = np.einsum("ti,tix->tx", potential[self.tri], self.grad)
        energy = self.area * np.einsum("tx,tx->t", field, field) * eps * EPS0 * self.factor
        bottom, top = energy[self.lower].sum(), energy[~self.lower].sum()
        if not surface:
            return bottom, top
        ctotal = bottom + top
        # Two physical half-domains; I=c*C*V single, I=c*(2*Cdiff)*V pair.
        current_capacitance = ctotal * (2 if self.g.coupled else 1)
        normal_e2 = np.einsum("tx,tx->t", field, field)[self.surface_tri] * 1e12
        surface_factor = 2 * np.dot(normal_e2, self.surface_length_m) * (EPS0 / current_capacitance) ** 2
        return bottom, top, surface_factor


@lru_cache(maxsize=8)
def _section(g: Geometry) -> _Section:
    return _Section(g)


def _array(value, frequency: np.ndarray, label: str, low: float, high: float) -> np.ndarray:
    try:
        a = np.broadcast_to(np.asarray(value, dtype=float), frequency.shape)
    except ValueError as error:
        raise ValueError(f"{label} must be a scalar or match the frequency vector.") from error
    if not np.all(np.isfinite(a)) or np.any(a < low) or np.any(a > high):
        raise ValueError(f"{label} must lie between {low:g} and {high:g}.")
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
            energies.append(section.solve(float(ratio), 1.0))
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


def analyze(geometry: Geometry, freq_hz, bottom_er=4.2, bottom_df=0.02,
            top_er=4.2, top_df=0.02, length_m=0.1, reference_ohm=50.0) -> Result:
    """Analyze a single or odd differential line; both ports use reference_ohm.

    impedance_ohm is Re(Zc) of lossy RLGC. quasistatic_impedance_ohm is
    sqrt(L_external/C), the geometric impedance often used for PCB synthesis.
    S11/S21 include propagation, finite length and mismatch at both ends.
    Return/insertion loss use positive -20 log10(abs(S)); matched S11 is capped
    at 300 dB for plotting. Frequency must be strictly increasing and >0 Hz.
    """
    geometry.validate()
    f = np.asarray(freq_hz, dtype=float)
    if (f.ndim != 1 or not len(f) or not np.all(np.isfinite(f))
            or np.any(f < 1e-3) or np.any(f > 1e15) or np.any(np.diff(f) <= 0)):
        raise ValueError("Frequencies must increase strictly within 0.001 to 1e15 Hz.")
    if not np.isfinite(length_m) or length_m < 0:
        raise ValueError("Line length must be finite and nonnegative.")
    if not np.isfinite(reference_ohm) or reference_ohm <= 0:
        raise ValueError("Port reference impedance must be finite and positive.")
    eb = _array(bottom_er, f, "Bottom permittivity", 1, 100)
    db = _array(bottom_df, f, "Bottom loss tangent", 0, 1)
    et = _array(top_er, f, "Top permittivity", 1, 100) if geometry.stripline else np.ones_like(f)
    dt = _array(top_df, f, "Top loss tangent", 0, 1) if geometry.stripline else np.zeros_like(f)
    section = _section(geometry)
    cb, ct = _capacitances(section, eb, et)
    capacitance = cb + ct
    cvac = sum(section.vacuum[:2])
    lexternal = 1 / (C0 ** 2 * cvac)
    omega = 2 * np.pi * f
    conductance = omega * (cb * db + ct * dt)
    area_m2 = (geometry.width_um + geometry.top_width_um) / 2 * geometry.thickness_um * 1e-12
    dcr = COPPER_RHO / area_m2
    rdc_mode = dcr * (2 if geometry.coupled else 1)
    surface_factor = section.vacuum[2]
    # shortcut: slab crossover estimates internal impedance; use magnetic diffusion for calibrated LF loss.
    depth = surface_factor * COPPER_RHO / rdc_mode
    z = np.sqrt(1j * omega * MU0 / COPPER_RHO) * depth
    internal = rdc_mode * z
    transition = z.real < 20
    internal[transition] /= np.tanh(z[transition])
    inductance = lexternal + internal.imag / omega
    resistance = internal.real
    series = resistance + 1j * omega * inductance
    shunt = conductance + 1j * omega * capacitance
    zc = np.sqrt(series / shunt)
    gamma = np.sqrt(series * shunt)
    # Decaying-wave formulation avoids cosh/sinh overflow on long/lossy lines.
    rho = (zc - reference_ohm) / (zc + reference_ohm)
    travel = np.exp(-gamma * length_m)
    denominator = 1 - rho ** 2 * travel ** 2
    s21 = (1 - rho ** 2) * travel / denominator
    s11 = rho * (1 - travel ** 2) / denominator
    attenuation = 20 / np.log(10) * gamma.real
    warnings = ["2D quasi-TEM estimate; smooth copper at 20 C; no roughness, vias or radiation.",
                "DC resistance is per signal conductor; ideal infinite return planes/rails have zero DC resistance."]
    dimension_m = (geometry.bottom_um + geometry.thickness_um +
                   (geometry.top_um if geometry.stripline else geometry.width_um)) * 1e-6
    if f[-1] * dimension_m * np.sqrt(max(eb.max(), et.max())) / C0 > 0.1:
        warnings.append("Cross-section exceeds one tenth wavelength; higher-order/full-wave validation required.")
    if max(db.max(), dt.max()) > 0.1:
        warnings.append("Dielectric loss uses a small-loss perturbation; tan(delta)>0.1 reduces accuracy.")
    active_dimensions = [geometry.width_um, geometry.thickness_um, geometry.bottom_um]
    if geometry.stripline:
        active_dimensions.append(geometry.top_um)
    if geometry.coupled:
        active_dimensions.append(geometry.gap_um)
    if geometry.shield:
        active_dimensions.append(geometry.clearance_um)
    if max(active_dimensions) / min(active_dimensions) > 100:
        warnings.append("Extreme dimension ratios: mesh accuracy needs independent convergence validation.")
    if not geometry.stripline:
        warnings.append("Exposed microstrip: all top-layer inputs are ignored; upper medium is air.")
    return Result(f.copy(), zc.real, -20 * np.log10(np.maximum(abs(s11), 1e-15)),
                  -20 * np.log10(np.maximum(abs(s21), 1e-300)), attenuation, dcr,
                  np.sqrt(lexternal / capacitance), zc, resistance, inductance,
                  conductance, capacitance, capacitance / cvac, s11, s21, gamma, tuple(warnings))


def synthesize_width(geometry: Geometry, target_ohm: float, frequency_hz: float,
                     bottom_er=4.2, bottom_df=0.02, top_er=4.2, top_df=0.02) -> float:
    """Return bottom width in um matching Re(lossy Zc) at one design frequency.

    The gap, thickness, height, angle and shield clearance remain fixed. The
    result is reanalyzed and must agree within 0.2% (or 0.05 ohm).
    """
    # A seed width need not be manufacturable: width is the quantity being solved.
    validation_width = max(4 * geometry.thickness_um, geometry.bottom_um, 1.0)
    replace(geometry, width_um=validation_width).validate()
    if not np.isfinite(target_ohm) or target_ohm <= 0:
        raise ValueError("Target impedance must be finite and positive.")
    inset = 2 * geometry.thickness_um / math.tan(math.radians(geometry.angle_deg))
    height = min(geometry.bottom_um, geometry.top_um) if geometry.stripline else geometry.bottom_um
    lo = max(inset * 1.001, height * 0.01, 0.01)
    hi = max(height * 30, inset * 2)

    def error(width):
        result = analyze(replace(geometry, width_um=float(width)), [frequency_hz],
                         bottom_er, bottom_df, top_er, top_df, length_m=0)
        return float(result.impedance_ohm[0] - target_ohm)

    elo, ehi = error(lo), error(hi)
    if elo * ehi > 0:
        raise ValueError(f"Target is outside the width-search range ({lo:.3g}–{hi:.3g} um; "
                         f"{ehi + target_ohm:.2f}–{elo + target_ohm:.2f} ohm).")
    width = brentq(error, lo, hi, xtol=max(1e-4, height * 1e-5), rtol=1e-5)
    if abs(error(width)) > max(0.05, target_ohm * 0.002):
        raise ValueError("Width synthesis did not meet the impedance tolerance.")
    return float(width)
