"""Physical/reference tests for the quasi-TEM engine (no external solver required)."""
import math
from dataclasses import replace
import time
import unittest

import numpy as np
from scipy.special import ellipk

from zcalc.engine import (C0, COPPER_RHO, COPPER_CONDUCTIVITY_S_M, EPS0, MU0, Geometry, KINDS, WARNINGS,
                          analyze, synthesize_width)


def hammerstad_jensen_zero_thickness(u, er):
    """H&J, IEEE MTT-S 1980 pp. 407–409, DOI 10.1109/MWSYM.1980.1124303."""
    a = 1 + math.log((u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432)) / 49
    a += math.log(1 + (u / 18.1) ** 3) / 18.7
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    effective = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    fu = 6 + (2 * math.pi - 6) * math.exp(-(30.666 / u) ** 0.7528)
    return math.sqrt(MU0 / EPS0) / (2 * math.pi * math.sqrt(effective)) * math.log(
        fu / u + math.sqrt(1 + (2 / u) ** 2))


def cohn_stripline(width, spacing, er, gap=None, even=False):
    """Exact zero-thickness stripline between planes `spacing` apart.

    Single strip: S. B. Cohn, IRE Trans. MTT 2(2), 1954, DOI 10.1109/TMTT.1954.1124875.
    With `gap`, the odd-mode (or even-mode) impedance of an edge-coupled pair:
    Cohn, IRE Trans. MTT 3(5), 1955, DOI 10.1109/TMTT.1955.1124973.
    """
    k = math.tanh(math.pi * width / (2 * spacing))
    if gap is not None:
        coupling = math.tanh(math.pi * (width + gap) / (2 * spacing))
        k = k * coupling if even else k / coupling
    return 30 * math.pi / math.sqrt(er) * ellipk(1 - k * k) / ellipk(k * k)


def receded(g, distance):
    """Geometry with every metal surface moved `distance` um into the metal."""
    angle = math.radians(g.angle_deg)
    edge = distance / math.sin(angle) + distance / math.tan(angle)
    changes = dict(width_um=g.width_um - 2 * edge, thickness_um=g.thickness_um - 2 * distance,
                   bottom_um=g.bottom_um + 2 * distance, top_um=g.top_um + 2 * distance)
    if g.coupled:
        changes["gap_um"] = g.gap_um + 2 * edge
    if g.shield:
        changes["clearance_um"] = g.clearance_um + edge + distance
    return replace(g, **changes)


def incremental_inductance(g):
    """Wheeler's R/Rs in 1/m from two separately meshed sections, receded and grown."""
    distance = 0.01 * min(g.top_width_um, *g.active_dimensions().values())
    inductance = [analyze(receded(g, d), [1e9], 1, 0, 1, 0).quasistatic_impedance_ohm[0] / C0
                  for d in (-distance, distance)]
    return (inductance[1] - inductance[0]) / (2 * distance * 1e-6) / MU0


def skin_resistance_factor(g):
    """R/Rs in 1/m from the engine, far above the skin-effect crossover."""
    frequency = 1e12
    surface_resistance = math.sqrt(math.pi * frequency * MU0 * COPPER_RHO)
    return analyze(g, [frequency], 1, 0, 1, 0).resistance_ohm_m[0] / surface_resistance


class EngineTests(unittest.TestCase):
    def test_default_conductivity_preserves_all_results(self):
        self.assertEqual(COPPER_CONDUCTIVITY_S_M, 1 / COPPER_RHO)
        for kind in KINDS:
            with self.subTest(kind=kind):
                g = Geometry(kind=kind)
                default = analyze(g, [1e6, 1e9, 1e12])
                explicit = analyze(g, [1e6, 1e9, 1e12], conductivity_s_m=COPPER_CONDUCTIVITY_S_M)
                for name, value in vars(default).items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(value, getattr(explicit, name), err_msg=name)
                    else:
                        self.assertEqual(value, getattr(explicit, name), name)

    def test_conductivity_dc_and_skin_scaling_all_modes_and_ports(self):
        f = [1e-3, 1e12]
        for kind in KINDS:
            with self.subTest(kind=kind):
                g = Geometry(kind=kind, shield=True, clearance_um=75)
                copper = analyze(g, f, 4, 0, 4, 0)
                poor = analyze(g, f, 4, 0, 4, 0, conductivity_s_m=COPPER_CONDUCTIVITY_S_M / 4)
                self.assertAlmostEqual(poor.dcr_ohm_m / copper.dcr_ohm_m, 4)
                self.assertAlmostEqual(poor.resistance_ohm_m[0] / copper.resistance_ohm_m[0], 4, places=7)
                self.assertAlmostEqual(poor.resistance_ohm_m[-1] / copper.resistance_ohm_m[-1], 2, places=7)
                np.testing.assert_array_equal(poor.capacitance_f_m, copper.capacitance_f_m)
                np.testing.assert_array_equal(poor.quasistatic_impedance_ohm, copper.quasistatic_impedance_ohm)
                self.assertGreater(poor.attenuation_db_m[-1], copper.attenuation_db_m[-1])
                self.assertFalse(np.array_equal(poor.s_parameters, copper.s_parameters))
                self.assertLessEqual(np.linalg.svd(poor.s_parameters, compute_uv=False).max(), 1 + 1e-12)
                if g.coupled:
                    even_copper = copper.even_impedance_ohm * copper.even_propagation_constant_m
                    even_poor = poor.even_impedance_ohm * poor.even_propagation_constant_m
                    self.assertAlmostEqual(even_poor.real[0] / even_copper.real[0], 4, places=7)
                    self.assertAlmostEqual(even_poor.real[-1] / even_copper.real[-1], 2, places=7)
                    s = poor.s_parameters
                    np.testing.assert_allclose((s[:, 0, 0] - s[:, 0, 2] - s[:, 2, 0] + s[:, 2, 2]) / 2,
                                               poor.s11, atol=1e-15)
                    np.testing.assert_allclose((s[:, 1, 0] - s[:, 1, 2] - s[:, 3, 0] + s[:, 3, 2]) / 2,
                                               poor.s21, atol=1e-15)

    def test_conductivity_extremes_are_finite_or_rejected_cleanly(self):
        # CMX uses a large finite conductivity for PEC; it is still a finite-loss metal here.
        for conductivity in (1, 5.8e12, 1e300):
            with self.subTest(conductivity=conductivity), np.errstate(over="raise", invalid="raise", divide="raise"):
                result = analyze(Geometry(kind="coupled_stripline"), [1e-3, 1e15],
                                 conductivity_s_m=conductivity)
                for value in vars(result).values():
                    if isinstance(value, np.ndarray):
                        self.assertTrue(np.all(np.isfinite(value)))
                self.assertGreater(result.dcr_ohm_m, 0)
        # A near-insulating finite conductivity can be represented for a large trace.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            poor = analyze(Geometry(width_um=1e5, thickness_um=1e5, bottom_um=1e5), [1e-3],
                           length_m=0, conductivity_s_m=1e-300)
        np.testing.assert_array_equal(poor.s11, 0)
        np.testing.assert_array_equal(poor.s21, 1)
        for invalid in (0, -1, float("nan"), float("inf"), float("-inf"), 1e-320,
                        [1e7], None, "invalid", 10 ** 1000):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "전도율"):
                analyze(Geometry(), [1e9], conductivity_s_m=invalid)
        with self.assertRaisesRegex(ValueError, "전도율"):
            analyze(Geometry(), [1e9], conductivity_s_m=1e-308)
        with self.assertRaisesRegex(ValueError, "전도율"):
            synthesize_width(Geometry(), 50, 1e9, conductivity_s_m=0)

    def test_width_synthesis_uses_selected_conductivity(self):
        conductivity = 1e6
        for kind in KINDS:
            with self.subTest(kind=kind):
                g = Geometry(kind=kind)
                target = 100 if g.coupled else 50
                width = synthesize_width(g, target, 1e9, 4, 0, 4, 0,
                                         conductivity_s_m=conductivity)
                result = analyze(replace(g, width_um=width), [1e9], 4, 0, 4, 0,
                                 conductivity_s_m=conductivity)
                self.assertLess(abs(result.impedance_ohm[0] - target), 0.05)
                wrong_material = analyze(replace(g, width_um=width), [1e9], 4, 0, 4, 0)
                self.assertGreater(abs(wrong_material.impedance_ohm[0] - target), 0.1)

    def test_thin_microstrip_matches_primary_published_reference(self):
        # 0.01 um approaches the zero-thickness H&J reference; <=2% is an
        # explicit numerical tolerance, not a general accuracy guarantee.
        for ratio in (0.2, 0.5, 1, 2, 5, 10):
            with self.subTest(width_height=ratio):
                g = Geometry(width_um=100 * ratio, thickness_um=0.01)
                actual = analyze(g, [1e9], 4, 0).quasistatic_impedance_ohm[0]
                expected = hammerstad_jensen_zero_thickness(ratio, 4)
                self.assertLess(abs(actual / expected - 1), 0.02)

    def test_thin_stripline_matches_exact_conformal_solutions(self):
        # 0.01 um copper approaches Cohn's zero-thickness strips. P1 elements
        # overestimate capacitance, so the impedance is slightly low.
        spacing = 200.0
        height = (spacing - 0.01) / 2
        for ratio in (0.1, 0.2, 0.5, 1, 2, 5):
            with self.subTest(width_spacing=ratio):
                g = Geometry(kind="stripline", width_um=ratio * spacing, thickness_um=0.01,
                             bottom_um=height, top_um=height)
                actual = analyze(g, [1e9], 4, 0, 4, 0).quasistatic_impedance_ohm[0]
                error = actual / cohn_stripline(ratio * spacing, spacing, 4) - 1
                self.assertTrue(-0.01 < error <= 0, error)
        for ratio, gap_ratio in ((0.2, 0.2), (0.5, 0.1), (0.5, 0.5), (1, 0.2), (1, 1)):
            with self.subTest(width_spacing=ratio, gap_spacing=gap_ratio):
                g = Geometry(kind="coupled_stripline", width_um=ratio * spacing, gap_um=gap_ratio * spacing,
                             thickness_um=0.01, bottom_um=height, top_um=height)
                actual = analyze(g, [1e9], 4, 0, 4, 0).quasistatic_impedance_ohm[0]
                expected = 2 * cohn_stripline(ratio * spacing, spacing, 4, gap_ratio * spacing)
                self.assertTrue(-0.01 < actual / expected - 1 <= 0, actual / expected - 1)
                # Even mode, per line. At 1 THz the internal inductance of the copper is negligible.
                even = abs(analyze(g, [1e12], 4, 0, 4, 0).even_impedance_ohm[0])
                expected = cohn_stripline(ratio * spacing, spacing, 4, gap_ratio * spacing, even=True)
                self.assertTrue(-0.01 < even / expected - 1 <= 0, even / expected - 1)

    def test_homogeneous_tem_scaling_and_parallel_plate_limit(self):
        g = Geometry(kind="stripline", width_um=10000, thickness_um=1)
        vacuum = analyze(g, [1e9], 1, 0, 1, 0)
        dielectric = analyze(g, [1e9], 4, 0, 4, 0)
        np.testing.assert_allclose(dielectric.effective_er, 4, rtol=1e-12)
        np.testing.assert_allclose(dielectric.quasistatic_impedance_ohm,
                                   vacuum.quasistatic_impedance_ohm / 2, rtol=1e-12)
        parallel_plate = math.sqrt(MU0 / EPS0) / 2 / (g.width_um / g.bottom_um + g.width_um / g.top_um)
        self.assertLess(abs(dielectric.quasistatic_impedance_ohm[0] / parallel_plate - 1), 0.015)
        np.testing.assert_allclose(vacuum.capacitance_f_m * (
            vacuum.quasistatic_impedance_ohm ** 2 * vacuum.capacitance_f_m), 1 / C0 ** 2, rtol=1e-12)

    def test_width_shields_gap_and_taper_physical_trends(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                g = Geometry(kind=kind)
                z = lambda geometry: analyze(geometry, [1e9], 4, 0.01, 3, 0.01).quasistatic_impedance_ohm[0]
                baseline = z(g)
                self.assertLess(z(replace(g, width_um=300)), baseline)
                self.assertLess(z(replace(g, shield=True, clearance_um=30)), baseline)
                self.assertGreater(z(replace(g, angle_deg=60)), baseline)
                if g.coupled:
                    self.assertGreater(z(replace(g, gap_um=300)), baseline)

    def test_independent_stripline_heights_and_dielectrics(self):
        g = Geometry(kind="stripline", bottom_um=80, top_um=160)
        base = analyze(g, [1e9], 4, 0, 2, 0)
        taller = analyze(replace(g, top_um=300), [1e9], 4, 0, 2, 0)
        high_er = analyze(g, [1e9], 4, 0, 6, 0)
        top_loss = analyze(g, [1e9], 4, 0, 2, 0.04)
        self.assertGreater(taller.quasistatic_impedance_ohm[0], base.quasistatic_impedance_ohm[0])
        self.assertLess(high_er.quasistatic_impedance_ohm[0], base.quasistatic_impedance_ohm[0])
        self.assertGreater(top_loss.conductance_s_m[0], 0)
        self.assertEqual(base.conductance_s_m[0], 0)
        swapped = analyze(replace(g, bottom_um=160, top_um=80), [1e9], 2, 0, 4, 0)
        np.testing.assert_allclose(base.quasistatic_impedance_ohm,
                                   swapped.quasistatic_impedance_ohm, rtol=0.004)

    def test_microstrip_top_inputs_ignored(self):
        a = analyze(Geometry(), [1e8, 1e9], 4, 0.01, 1, 0)
        b = analyze(Geometry(top_um=-1), [1e8, 1e9], 4, 0.01, [99, 100], [0.8, 1])
        np.testing.assert_array_equal(a.impedance_ohm, b.impedance_ohm)
        np.testing.assert_array_equal(a.conductance_s_m, b.conductance_s_m)

    def test_dc_area_and_skin_loss(self):
        g = Geometry(angle_deg=60)
        result = analyze(g, [1, 1e8, 1e9, 1e10], 4, 0)
        area = (g.width_um + g.top_width_um) / 2 * g.thickness_um * 1e-12
        self.assertAlmostEqual(result.dcr_ohm_m, COPPER_RHO / area, places=12)
        self.assertAlmostEqual(result.resistance_ohm_m[0] / result.dcr_ohm_m, 1, places=5)
        self.assertTrue(np.all(np.diff(result.resistance_ohm_m) > 0))
        self.assertAlmostEqual(result.resistance_ohm_m[-1] / result.resistance_ohm_m[-2], math.sqrt(10), places=4)
        pair = analyze(replace(g, kind="coupled_microstrip"), [1], 4, 0)
        self.assertAlmostEqual(pair.resistance_ohm_m[0] / result.dcr_ohm_m, 2, places=5)
        self.assertEqual(pair.dcr_ohm_m, result.dcr_ohm_m)

    def test_skin_resistance_matches_incremental_inductance(self):
        # Wheeler's rule R = (Rs/mu0) dL/dn, evaluated independently by solving
        # separately meshed cross-sections whose metal surfaces recede by n.
        # The former surface-field integral was 9-33% low on these sections.
        for g in (Geometry(width_um=180, thickness_um=18),
                  Geometry(kind="stripline", width_um=80, thickness_um=30, bottom_um=60,
                           top_um=140, angle_deg=50),
                  Geometry(kind="coupled_microstrip", gap_um=30, angle_deg=70),
                  Geometry(kind="coupled_stripline", thickness_um=18, gap_um=120,
                           shield=True, clearance_um=150)):
            with self.subTest(g=g):
                self.assertLess(abs(skin_resistance_factor(g) / incremental_inductance(g) - 1), 0.01)

    def test_skin_resistance_with_gaps_far_thinner_than_copper(self):
        # The receding distance of the derivative must stay small against every gap,
        # not only against the copper: these sections were up to 100% off otherwise.
        for g in (Geometry(width_um=200, thickness_um=35, bottom_um=0.1),
                  Geometry(width_um=3000, thickness_um=2000, bottom_um=1),
                  Geometry(kind="coupled_microstrip", width_um=60, thickness_um=50, bottom_um=5, gap_um=0.6),
                  Geometry(kind="stripline", width_um=150, thickness_um=30, bottom_um=0.5, top_um=200,
                           shield=True, clearance_um=0.4)):
            with self.subTest(g=g):
                self.assertLess(abs(skin_resistance_factor(g) / incremental_inductance(g) - 1), 0.01)
        # Parallel-plate limit: strip and plane each contribute Rs / W.
        plates = Geometry(width_um=200, thickness_um=35, bottom_um=0.1)
        self.assertLess(abs(skin_resistance_factor(plates) * plates.width_um * 1e-6 / 2 - 1), 0.02)

    def test_stripline_skin_resistance_matches_published_formula(self):
        # D. M. Pozar, Microwave Engineering, 4th ed., sec. 3.7, wide-strip branch
        # (sqrt(er) Z0 < 120 ohm): alpha_c = 2.7e-3 Rs er Z0 A / (30 pi (b - t)) Np/m,
        # A = 1 + 2W/(b-t) + (b+t)/(pi (b-t)) ln((2b-t)/t), and R = 2 Z0 alpha_c.
        # The closed form is itself approximate; 3% is the comparison tolerance.
        for width, thickness, spacing in ((120, 18, 218), (200, 18, 318), (300, 17, 217)):
            with self.subTest(width=width, thickness=thickness, spacing=spacing):
                height = (spacing - thickness) / 2
                g = Geometry(kind="stripline", width_um=width, thickness_um=thickness,
                             bottom_um=height, top_um=height)
                z0 = analyze(g, [1e9], 1, 0, 1, 0).quasistatic_impedance_ohm[0]
                self.assertLess(z0, 120)
                a = (1 + 2 * width / (spacing - thickness) + (spacing + thickness)
                     / (math.pi * (spacing - thickness)) * math.log((2 * spacing - thickness) / thickness))
                expected = 2 * z0 * 2.7e-3 * z0 * a / (30 * math.pi * (spacing - thickness) * 1e-6)
                self.assertLess(abs(skin_resistance_factor(g) / expected - 1), 0.03)

    def test_differential_definition_approaches_twice_single(self):
        single = analyze(Geometry(), [1e9], 4, 0)
        pair = analyze(Geometry(kind="coupled_microstrip", gap_um=10000), [1e9], 4, 0)
        ratio = pair.quasistatic_impedance_ohm[0] / single.quasistatic_impedance_ohm[0]
        self.assertLess(abs(ratio / 2 - 1), 0.015)

    def test_scattering_matches_abcd_and_is_passive(self):
        f = np.geomspace(1e4, 40e9, 151)
        for kind in KINDS:
            with self.subTest(kind=kind):
                result = analyze(Geometry(kind=kind), f, 4, 0.02, 3.5, 0.01,
                                 length_m=0.07, reference_ohm=73)
                z, gl = result.complex_impedance_ohm, result.propagation_constant_m * 0.07
                a, b, c = np.cosh(gl), z * np.sinh(gl), np.sinh(gl) / z
                denominator = 2 * a + b / 73 + c * 73
                np.testing.assert_allclose(result.s21, 2 / denominator, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(result.s11, (b / 73 - c * 73) / denominator,
                                           rtol=1e-11, atol=1e-12)
                self.assertTrue(np.all(abs(result.s11) ** 2 + abs(result.s21) ** 2 <= 1 + 1e-12))
                self.assertTrue(np.all(result.attenuation_db_m > 0))

    def test_s_parameters_in_db_are_not_positive(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                result = analyze(Geometry(kind=kind), np.geomspace(1e6, 4e10, 60), 4, 0.02, 3.5, 0.01,
                                 length_m=0.1, reference_ohm=60)
                np.testing.assert_allclose(result.s11_db, 20 * np.log10(abs(result.s11)))
                np.testing.assert_allclose(result.s21_db, 20 * np.log10(abs(result.s21)))
                self.assertTrue(np.all(result.s11_db < 0) and np.all(result.s21_db < 0))

    def test_single_ended_ports_of_line_and_pair(self):
        f = np.linspace(1e8, 2e10, 80)
        single = analyze(Geometry(), f, 4.2, 0.018, length_m=0.05, reference_ohm=50)
        self.assertEqual((single.s_parameters.shape, single.port_ohm), ((80, 2, 2), 50))
        np.testing.assert_array_equal(single.s_parameters[:, 1, 0], single.s21)
        np.testing.assert_array_equal(single.s_parameters[:, 0, 0], single.s_parameters[:, 1, 1])
        self.assertIsNone(single.even_impedance_ohm)
        self.assertIsNone(analyze(Geometry(), f, ports=False).s_parameters)
        for kind in ("coupled_microstrip", "coupled_stripline"):
            with self.subTest(kind=kind):
                g = Geometry(kind=kind, width_um=120, gap_um=90, thickness_um=30, angle_deg=75)
                pair = analyze(g, f, 4.1, 0.02, 3.6, 0.01, length_m=0.08, reference_ohm=100)
                s = pair.s_parameters
                self.assertEqual((s.shape, pair.port_ohm), ((80, 4, 4), 50))
                np.testing.assert_array_equal(s, s.transpose(0, 2, 1))  # reciprocal
                self.assertLessEqual(np.linalg.svd(s, compute_uv=False).max(), 1)  # passive
                # Ports 1/3 are the near-end pair and 2/4 the far-end pair: the differential
                # combination recovers the differential two-port.
                np.testing.assert_allclose((s[:, 0, 0] - s[:, 0, 2] - s[:, 2, 0] + s[:, 2, 2]) / 2, pair.s11, atol=1e-15)
                np.testing.assert_allclose((s[:, 1, 0] - s[:, 1, 2] - s[:, 3, 0] + s[:, 3, 2]) / 2, pair.s21, atol=1e-15)
                self.assertTrue(np.all(pair.even_impedance_ohm.real > pair.complex_impedance_ohm.real / 2))
        # Far apart, each line of a pair behaves as the single line and nothing couples.
        apart = analyze(Geometry(kind="coupled_microstrip", gap_um=20000), f, 4.2, 0.018,
                        length_m=0.05, reference_ohm=100)
        np.testing.assert_allclose(apart.even_impedance_ohm, single.complex_impedance_ohm, rtol=0.005)
        # (Magnitudes: a 0.2% difference in velocity between two grids adds up in phase.)
        np.testing.assert_allclose(abs(apart.s_parameters[:, 1, 0]), abs(single.s21), rtol=0.01)
        self.assertLess(abs(apart.s_parameters[:, 2:, 0]).max(), 1e-3)
        # A homogeneous dielectric gives both modes one velocity. Microstrip does not: its even
        # mode is slower, which is what makes its far-end crosstalk large.
        strip, micro = (analyze(Geometry(kind=kind, width_um=120, gap_um=90, thickness_um=18), f, 4, 0.01, 4, 0.01,
                                length_m=0.08, reference_ohm=100) for kind in ("coupled_stripline", "coupled_microstrip"))
        slowness = lambda r: (r.even_propagation_constant_m.imag / r.propagation_constant_m.imag)[-1]
        self.assertAlmostEqual(slowness(strip), 1, delta=0.005)
        self.assertGreater(slowness(micro), 1.05)
        self.assertLess(abs(strip.s_parameters[:, 3, 0]).max(), 0.1 * abs(micro.s_parameters[:, 3, 0]).max())

    def test_zero_length_and_very_long_line_stay_finite(self):
        zero = analyze(Geometry(), [1e6, 1e9], length_m=0)
        np.testing.assert_array_equal(zero.s11, 0)
        np.testing.assert_allclose(zero.s21, 1, atol=2e-15)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            long = analyze(Geometry(thickness_um=1000), [1e-3, 1e15], length_m=1e6)
        for output in (long.s11, long.s21, long.s21_db, long.s11_db, long.s_parameters):
            self.assertTrue(np.all(np.isfinite(output)))
        self.assertEqual(long.s21[-1], 0)

    def test_material_sweep_interpolation_matches_direct_solves(self):
        f = np.geomspace(1e6, 20e9, 81)
        eb = np.geomspace(1, 100, len(f))
        et = eb[::-1]
        df = np.linspace(0.005, 0.03, len(f))
        g = Geometry(kind="coupled_stripline")
        sweep = analyze(g, f, eb, df, et, df / 2)
        for i in (1, 9, 19, 39, 59, 71, 79):
            direct = analyze(g, [f[i]], eb[i], df[i], et[i], df[i] / 2)
            self.assertLess(abs(sweep.capacitance_f_m[i] / direct.capacitance_f_m[0] - 1), 0.001)
            self.assertLess(abs(sweep.conductance_s_m[i] / direct.conductance_s_m[0] - 1), 0.001)

    def test_width_synthesis_reanalyzes_all_four_modes(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                g = Geometry(kind=kind, shield=True, clearance_um=150, angle_deg=70)
                target = 85 if g.coupled else 45
                width = synthesize_width(g, target, 1e9, 4, 0.02, 3, 0.01)
                actual = analyze(replace(g, width_um=width), [1e9], 4, 0.02, 3, 0.01)
                self.assertLess(abs(actual.impedance_ohm[0] - target), 0.05)

    def test_far_shield_and_unmanufacturable_synthesis_seed(self):
        # Previously a shield beyond the open-domain boundary reversed the mesh.
        for kind in ("microstrip", "coupled_stripline"):
            g = Geometry(kind=kind)
            baseline = analyze(g, [1e9], 4, 0, 4, 0)
            far = analyze(replace(g, shield=True, clearance_um=100000), [1e9], 4, 0, 4, 0)
            self.assertLess(abs(far.quasistatic_impedance_ohm[0] / baseline.quasistatic_impedance_ohm[0] - 1), 0.015)
        g = Geometry(width_um=10, thickness_um=35, angle_deg=60)
        width = synthesize_width(g, 50, 1e9)
        self.assertGreater(width, 2 * g.thickness_um / math.tan(math.radians(g.angle_deg)))
        self.assertLess(abs(analyze(replace(g, width_um=width), [1e9]).impedance_ohm[0] - 50), 0.05)

    def test_invalid_inputs(self):
        for f in ([], [0], [-1], [1e-4], [1e16], [1, 1], [2, 1], [np.nan], [[1, 2]]):
            with self.subTest(f=f), self.assertRaises(ValueError):
                analyze(Geometry(), f)
        for g in (Geometry(width_um=-1), Geometry(thickness_um=0), Geometry(bottom_um=1e-10),
                  Geometry(kind="unknown"), Geometry(angle_deg=40),
                  Geometry(width_um=10, angle_deg=45), Geometry(shield=True, clearance_um=0)):
            with self.subTest(g=g), self.assertRaises(ValueError):
                analyze(g, [1e9])
        for kwargs in ({"bottom_er": 0.9}, {"bottom_df": -0.1},
                       {"bottom_er": [2, 3]}, {"length_m": -1}, {"reference_ohm": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze(Geometry(), [1e9], **kwargs)
        with self.assertRaises(ValueError):
            synthesize_width(Geometry(), 10000, 1e9)

    def test_error_messages_name_the_input_and_warnings_use_known_keys(self):
        with self.assertRaisesRegex(ValueError, "도선 두께 T"):
            analyze(Geometry(thickness_um=0), [1e9])
        with self.assertRaisesRegex(ValueError, "실드 간격 G"):
            analyze(Geometry(shield=True, clearance_um=float("nan")), [1e9])
        with self.assertRaisesRegex(ValueError, "하부 유전체 Dk"):
            analyze(Geometry(), [1e9], bottom_er=0.5)
        self.assertEqual(analyze(Geometry(), [1e9]).warnings, ())
        flagged = analyze(Geometry(width_um=20000), [1e9, 40e9], 4, 0.2).warnings
        self.assertEqual(set(flagged), set(WARNINGS))


if __name__ == "__main__":
    start = time.perf_counter()
    unittest.main()
