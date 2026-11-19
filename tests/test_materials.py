"""Synthetic CMX fixtures only; user material libraries stay local."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from zcalc.materials import Conductor, load_cmx, load_material_library, manual_material


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
        with self.assertRaisesRegex(ValueError, "주파수 단위"):
            self.load('<Material name="Bad unit"><Dielectric><Model>1 4 .01</Model></Dielectric></Material>', "rpm")
        with self.assertRaisesRegex(ValueError, "유전체 모델 데이터가 없습니다"):
            self.load('<Material name="Empty"><Dielectric><Model/></Dielectric></Material>')

    def test_dtd_and_wrong_root_are_rejected(self):
        for xml in ['<!DOCTYPE other [<!ENTITY x "4">]><Cadence_Material_Lib/>', '<other/>', '<broken>']:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.cmx"
                path.write_text(xml, encoding="utf-8")
                with self.subTest(xml=xml), self.assertRaises(ValueError):
                    load_cmx(path)


class ConductorLibraryTests(unittest.TestCase):
    def load(self, records, unit="S/m", temperature_unit="C", encoding="utf-8", loader=load_material_library):
        declaration = "utf-8" if encoding == "utf-8-sig" else encoding
        xml = f'''<?xml version="1.0" encoding="{declaration}"?>
        <Cadence_Material_Lib Version="1.05"><DataDescriptions>
        <Metal><Column name="Temperature" unit="{temperature_unit}"/>
        <Column name="Conductivity" unit="{unit}"/></Metal>
        </DataDescriptions>{records}</Cadence_Material_Lib>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "재료 library with spaces.cmx"
            path.write_bytes(xml.encode(encoding))
            return loader(path)

    def test_mixed_file_reads_and_parses_once(self):
        records = '''<Material name="혼합"><Dielectric><Model>100 4 0.01</Model></Dielectric>
        <Metal><Model>40 4e7\n10 7e7\n10 7e7</Model></Metal></Material>'''
        with patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as reader:
            with patch("zcalc.materials.ET.fromstring", wraps=ET.fromstring) as parser:
                dielectrics, conductors = self.load(records)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(parser.call_count, 1)
        self.assertEqual(list(dielectrics), ["혼합"])
        self.assertEqual(list(conductors), ["혼합"])
        self.assertAlmostEqual(conductors["혼합"].conductivity_s_m, 6e7)
        self.assertEqual(conductors["혼합"].temperature_c, 20)
        np.testing.assert_allclose(dielectrics["혼합"].sample(1e8)[0], 4)

    def test_single_kind_libraries_and_legacy_loader(self):
        metal = '<Material name="Only metal"><Metal><Model>25 3e7</Model></Metal></Material>'
        dielectrics, conductors = self.load(metal)
        self.assertEqual(dielectrics, {})
        self.assertEqual(conductors["Only metal"].temperature_c, 25)
        with self.assertRaisesRegex(ValueError, "유전체 모델 데이터가 없습니다"):
            self.load(metal, loader=load_cmx)
        dielectric = '<Material name="Only dielectric"><Dielectric><Model>100 3 .02</Model></Dielectric></Material>'
        dielectrics, conductors = self.load(dielectric)
        self.assertEqual(list(dielectrics), ["Only dielectric"])
        self.assertEqual(conductors, {})
        malformed_metal = '<Material name="Ignored metal"><Metal><Model>nan -1</Model></Metal></Material>'
        self.assertEqual(list(self.load(dielectric + malformed_metal, loader=load_cmx)), ["Only dielectric"])
        with self.assertRaises(ValueError):
            self.load(dielectric + malformed_metal)
        with self.assertRaisesRegex(ValueError, "유전체나 도체 모델 데이터가 없습니다"):
            self.load('<Material name="Blank"><Metal><Model> </Model></Metal></Material>')

    def test_temperature_clamping_constants_and_tagged_values(self):
        _, materials = self.load('''
        <Material name="Below"><Metal><Model>30 3e7\n50 2e7</Model></Metal></Material>
        <Material name="Above"><Metal><Model>-10 8e7\n10 6e7</Model></Metal></Material>
        <Material name="Constant"><Metal><Model>4e7</Model></Metal></Material>
        <Material name="Tagged"><Metal><Model Temperature="10">7e7</Model>
        <Model Temperature="30">5e7</Model></Metal></Material>
        <Material name="PEC"><Metal><Model>20 1e13</Model></Metal></Material>''')
        self.assertEqual(materials["Below"], Conductor("Below", 3e7, 30))
        self.assertEqual(materials["Above"], Conductor("Above", 6e7, 10))
        self.assertEqual(materials["Constant"], Conductor("Constant", 4e7, None))
        self.assertEqual(materials["Tagged"], Conductor("Tagged", 6e7, 20))
        self.assertEqual(materials["PEC"].conductivity_s_m, 1e13)

    def test_conductivity_units_and_windows_encodings(self):
        record = '<Material name="도체"><Metal><Model>20 2</Model></Metal></Material>'
        for unit, expected in [("S/m", 2), ("S/cm", 200), ("S/mm", 2000), ("MS/m", 2e6), ("mS/m", .002)]:
            with self.subTest(unit=unit):
                _, conductors = self.load(record, unit=unit)
                self.assertAlmostEqual(conductors["도체"].conductivity_s_m, expected)
        for encoding in ["utf-8-sig", "utf-16", "utf-16-be", "utf-16-le"]:
            with self.subTest(encoding=encoding):
                _, conductors = self.load(record, encoding=encoding)
                self.assertEqual(conductors["도체"].conductivity_s_m, 2)
        for unit in ["Ohm*m", "S/in", "s/m"]:
            with self.subTest(unit=unit), self.assertRaisesRegex(ValueError, "전도율 단위"):
                self.load(record, unit=unit)
        with self.assertRaisesRegex(ValueError, "섭씨"):
            self.load(record, temperature_unit="K")

    def test_invalid_metal_tables_fail_atomically(self):
        for rows in ["20 nan", "nan 2e7", "inf 2e7", "20 inf", "20 0", "20 -1", "20 3e7\n20 4e7",
                     "20 3e7 0", "3e7\n20 3e7", "3e7\n4e7"]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load(f'<Material name="Invalid"><Metal><Model>{rows}</Model></Metal></Material>')
        for attributes in ['Temperature="nan"', 'Temperature="inf"']:
            with self.subTest(attributes=attributes), self.assertRaises(ValueError):
                self.load(f'<Material name="Bad temperature"><Metal><Model {attributes}>3e7</Model></Metal></Material>')
        for names in [('duplicate', 'duplicate'), ('', 'valid')]:
            records = ''.join(f'<Material name="{name}"><Metal><Model>20 3e7</Model></Metal></Material>' for name in names)
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "이름"):
                self.load(records)
        for conductivity in [0, -1, float("inf"), float("nan")]:
            with self.subTest(conductivity=conductivity), self.assertRaises(ValueError):
                Conductor("Invalid", conductivity)

    def test_utf16_dtd_rejected_for_combined_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.cmx"
            path.write_bytes('<!DOCTYPE other [<!ENTITY x "4">]><Cadence_Material_Lib/>'.encode("utf-16"))
            with self.assertRaisesRegex(ValueError, "DTD"):
                load_material_library(path)

    def test_malformed_encodings_are_import_errors(self):
        for data in [b'<?xml version="1.0" encoding="unknown-codec"?><Cadence_Material_Lib/>', b'\xff\xfe\x00']:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "encoding.cmx"
                path.write_bytes(data)
                with self.subTest(data=data), self.assertRaisesRegex(ValueError, "XML을 읽을 수 없습니다"):
                    load_material_library(path)


if __name__ == "__main__":
    unittest.main()
