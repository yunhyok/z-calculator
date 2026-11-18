"""Physical/reference tests for the quasi-TEM engine (no external solver required)."""
import math
from dataclasses import replace
import time
import unittest

import numpy as np

from zcalc.engine import (C0, COPPER_RHO, EPS0, MU0, Geometry, KINDS,
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


class EngineTests(unittest.TestCase):
    def test_thin_microstrip_matches_primary_published_reference(self):
        # 0.01 um approaches the zero-thickness H&J reference; <=2% is an
        # explicit numerical tolerance, not a general accuracy guarantee.
        for ratio in (0.2, 0.5, 1, 2, 5, 10):
            with self.subTest(width_height=ratio):
                g = Geometry(width_um=100 * ratio, thickness_um=0.01)
                actual = analyze(g, [1e9], 4, 0).quasistatic_impedance_ohm[0]
                expected = hammerstad_jensen_zero_thickness(ratio, 4)
                self.assertLess(abs(actual / expected - 1), 0.02)

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

    def test_zero_length_and_very_long_line_stay_finite(self):
        zero = analyze(Geometry(), [1e6, 1e9], length_m=0)
        np.testing.assert_array_equal(zero.s11, 0)
        np.testing.assert_allclose(zero.s21, 1, atol=2e-15)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            long = analyze(Geometry(thickness_um=1000), [1e-3, 1e15], length_m=1e6)
        for output in (long.s11, long.s21, long.insertion_loss_db, long.return_loss_db):
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


if __name__ == "__main__":
    start = time.perf_counter()
    unittest.main()
