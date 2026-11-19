"""Touchstone 1.0 layout, checked with a parser written from the specification."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from zcalc.engine import Geometry, analyze
from zcalc.touchstone import PORT_ORDERS, format_touchstone, save_touchstone


def parse(text):
    """Option line, comments and (frequency, matrix) per point of a Version 1.0 file.

    Touchstone File Format Specification 2.0, rules for Version 1.0 files: at most
    four pairs per line, the frequency only at the start of a point, two-port
    data ordered 11 21 12 22 and larger matrices row by row.
    """
    comments, option, numbers = [], None, []
    for line in text.splitlines():
        line, _, comment = line.partition("!")
        if comment:
            comments.append(comment.strip())
        if line.startswith("#"):
            option = line.split()
        elif line.strip():
            values = [float(value) for value in line.split()]
            assert len(values) in (8, 9), "more than four pairs on a data line"
            numbers.append(values)
    ports = 2 if all(len(values) == 9 for values in numbers) else 4
    points = []
    for start in range(0, len(numbers), 1 if ports == 2 else ports):
        rows = numbers[start:start + (1 if ports == 2 else ports)]
        assert len(rows[0]) == 9 and all(len(row) == 8 for row in rows[1:])
        flat = np.array(rows[0][1:] + [value for row in rows[1:] for value in row])
        matrix = (flat[0::2] + 1j * flat[1::2]).reshape(ports, ports)
        points.append((rows[0][0], matrix.T if ports == 2 else matrix))
    return option, comments, points


class TouchstoneTests(unittest.TestCase):
    def test_two_port_file_lists_s21_before_s12(self):
        frequency = [1e9, 2e9, 3e9]
        matrix = np.array([[0.1 + 0.2j, 0.3 - 0.4j], [0.5 + 0.6j, -0.7 + 0.8j]])  # S12 differs from S21
        s = np.stack([matrix * scale for scale in (1, 0.5, 0.25)])
        text = format_touchstone(frequency, s, 75, ["first", "second"])
        option, comments, points = parse(text)
        self.assertEqual(option, ["#", "Hz", "S", "RI", "R", "75"])
        self.assertEqual(comments[:4], ["first", "second", "Port 1: near end", "Port 2: far end"])
        self.assertEqual([f for f, _ in points], frequency)
        np.testing.assert_allclose([m for _, m in points], s, rtol=1e-12)
        first = [float(value) for value in next(line for line in text.splitlines() if line[0] not in "!#").split()]
        self.assertEqual(first[3:5], [0.5, 0.6])  # the second pair is S21

    def test_four_port_file_is_row_wise_in_either_port_order(self):
        f = np.linspace(1e8, 2e10, 25)
        result = analyze(Geometry(kind="coupled_microstrip", gap_um=80), f, 4.1, 0.02, length_m=0.06,
                         reference_ohm=100)
        s = result.s_parameters
        for order, index in PORT_ORDERS.items():
            with self.subTest(order=order):
                option, comments, points = parse(format_touchstone(f, s, result.port_ohm, order=order))
                self.assertEqual(option[-1], "50")
                self.assertEqual(len(points), len(f))
                np.testing.assert_allclose([m for _, m in points], s[:, index][:, :, index], rtol=1e-11, atol=1e-13)
                np.testing.assert_allclose([frequency for frequency, _ in points], f, rtol=1e-12)
        # "1-2": lines run 1 to 2 and 3 to 4. "1-3": the near end is ports 1 and 2.
        through = abs(s[10, 1, 0])
        _, comments, points = parse(format_touchstone(f, s, 50, order="1-3"))
        self.assertAlmostEqual(abs(points[10][1][2, 0]), through)
        self.assertEqual(comments[:5], ["Port 1: line + (near end)", "Port 2: line - (near end)",
                                        "Port 3: line + (far end)", "Port 4: line - (far end)",
                                        "Differential pairs: ports 1 and 2 (near end), ports 3 and 4 (far end)"])
        # Reading that file with the pairs it names gives the differential two-port. Pairing
        # the ports as the other layout does (ports 1 and 3, 2 and 4) gives something else.
        matrix = np.array([m for _, m in points])
        differential = lambda m, near, far: (m[:, far[0], near[0]] - m[:, far[0], near[1]]
                                             - m[:, far[1], near[0]] + m[:, far[1], near[1]]) / 2
        np.testing.assert_allclose(differential(matrix, (0, 1), (2, 3)), result.s21, rtol=1e-10, atol=1e-13)
        self.assertGreater(abs(differential(matrix, (0, 2), (1, 3)) - result.s21).max(), 0.5)

    def test_files_are_ascii_and_reject_unusable_data(self):
        s = analyze(Geometry(), [1e9, 2e9], length_m=0.01).s_parameters
        text = format_touchstone([1e9, 2e9], s, 50, ["재료 FR-4 µ\n# second line of the material name"])
        self.assertTrue(text.isascii())
        self.assertIn("FR-4", text)
        self.assertIn("! # second line of the material name", text.splitlines())
        for arguments in (([1e9], s, 50), ([2e9, 1e9], s, 50), ([1e9, 2e9], s[:, :1, :1], 50),
                          ([1e9, 2e9], np.full_like(s, np.nan), 50)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                format_touchstone(*arguments)
        with self.assertRaises(ValueError):
            format_touchstone([1e9, 2e9], np.zeros((2, 4, 4)), 50, order="2-1")

    def test_save_replaces_the_file_only_on_success(self):
        s = analyze(Geometry(), [1e9, 2e9], length_m=0.01).s_parameters
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "line.s2p"
            save_touchstone(path, [1e9, 2e9], s, 50, ["saved"])
            saved = path.read_text(encoding="ascii")
            self.assertEqual(saved, format_touchstone([1e9, 2e9], s, 50, ["saved"]))
            with patch("zcalc.touchstone.os.replace", side_effect=OSError("simulated write error")):
                with self.assertRaises(OSError):
                    save_touchstone(path, [1e9, 2e9], s, 75)
            with self.assertRaises(ValueError):
                save_touchstone(path, [2e9, 1e9], s, 75)
            self.assertEqual(path.read_text(encoding="ascii"), saved)
            self.assertEqual([item.name for item in Path(folder).iterdir()], ["line.s2p"])


if __name__ == "__main__":
    unittest.main()
