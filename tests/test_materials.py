"""Synthetic CMX fixtures only; user material libraries stay local."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from zcalc.materials import load_cmx, manual_material


class MaterialsTests(unittest.TestCase):
    def load(self, material_xml, unit="MHz"):
        xml = f'''<?xml version="1.0"?><Cadence_Material_Lib Version="1.05">
        <DataDescriptions><Dielectric>
        <Column name="Temperature" unit="C"/><Column name="Frequency" unit="{unit}"/>
        <Column name="Relative Permittivity"/><Column name="LossTangent"/>
        </Dielectric></DataDescriptions>{material_xml}</Cadence_Material_Lib>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.cmx"
            path.write_text(xml, encoding="utf-8")
            return load_cmx(path)

    def test_frequency_tables_and_endpoint_hold(self):
        material = self.load('''<Material name="Synthetic"><Dielectric><Model>
            100 3.0 0.03
            1 4.0 0.01
            1 4.0 0.01
            </Model></Dielectric></Material>''')["Synthetic"]
        er, df = material.sample([1e5, 1e6, 1e7, 1e8, 1e9])
        np.testing.assert_allclose(er, [4, 4, 3.5, 3, 3])
        np.testing.assert_allclose(df, [.01, .01, .02, .03, .03])
        self.assertEqual(material.frequency_range_hz, (1e6, 1e8))
        self.assertFalse(material.is_constant)

    def test_constant_and_non_dielectric_records(self):
        materials = self.load('''<Material name="Metal"><Metal><Model>25 1e7</Model></Metal></Material>
        <Material name="Blank"><Dielectric><Model> </Model></Dielectric></Material>
        <Material name="Constant"><Dielectric><Model>2 4.2 0.02</Model></Dielectric></Material>''', "GHz")
        self.assertEqual(list(materials), ["Constant"])
        material = materials["Constant"]
        self.assertTrue(material.is_constant)
        self.assertEqual(material.frequency_range_hz, (2e9, 2e9))
        er, df = material.sample([[1e6, 1e12]])
        np.testing.assert_allclose(er, [[4.2, 4.2]])
        np.testing.assert_allclose(df, [[.02, .02]])

    def test_default_stratum_and_nearest_temperature(self):
        materials = self.load('''<Material name="Default"><Dielectric>
        <Model Temperature="25">10 7 0.03</Model><Model>10 4 0.01</Model>
        </Dielectric></Material><Material name="Nearest"><Dielectric>
        <Model Temperature="40">10 6 0.03</Model><Model Temperature="10">10 5 0.02</Model>
        </Dielectric></Material>''')
        self.assertEqual(materials["Default"].sample(1e7)[0], 4)
        self.assertIsNone(materials["Default"].temperature_c)
        self.assertEqual(materials["Nearest"].sample(1e7)[0], 5)
        self.assertEqual(materials["Nearest"].temperature_c, 10)

    def test_manual_and_invalid_input(self):
        material = manual_material(4.0, 0)
        np.testing.assert_allclose(material.sample([1e6, 1e10])[0], [4, 4])
        self.assertIsInstance(material.sample(1e9)[0], np.ndarray)
        for er, df in [(0, 0), (4, -1), (float("nan"), .01), (4, float("inf"))]:
            with self.subTest(er=er, df=df), self.assertRaises(ValueError):
                manual_material(er, df)
        for frequency in [0, -1, float("nan"), float("inf")]:
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                material.sample(frequency)

    def test_invalid_populated_model_fails_import(self):
        for rows in ["1 4", "1 nan 0.01", "0 4 0.01", "1 4 -0.01", "1 4 .01\n1 5 .01"]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load(f'<Material name="Invalid"><Dielectric><Model>{rows}</Model></Dielectric></Material>')
        with self.assertRaisesRegex(ValueError, "frequency unit"):
            self.load('<Material name="Bad unit"><Dielectric><Model>1 4 .01</Model></Dielectric></Material>', "rpm")
        with self.assertRaisesRegex(ValueError, "No populated"):
            self.load('<Material name="Empty"><Dielectric><Model/></Dielectric></Material>')

    def test_dtd_and_wrong_root_are_rejected(self):
        for xml in ['<!DOCTYPE other [<!ENTITY x "4">]><Cadence_Material_Lib/>', '<other/>', '<broken>']:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.cmx"
                path.write_text(xml, encoding="utf-8")
                with self.subTest(xml=xml), self.assertRaises(ValueError):
                    load_cmx(path)


if __name__ == "__main__":
    unittest.main()
