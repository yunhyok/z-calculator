"""The release version is repeated in the packaging inputs; they must agree."""

from pathlib import Path
import re
import unittest

from zcalc import __version__

ROOT = Path(__file__).resolve().parent.parent


class ReleaseTests(unittest.TestCase):
    def test_packaging_inputs_carry_the_package_version(self):
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")
        read = lambda name: (ROOT / name).read_text(encoding="utf-8")
        installer = f"Z-Calculator-{__version__}-Windows-x64-Setup.exe"
        numbers = ", ".join(__version__.split(".")) + ", 0"
        expected = {
            "packaging/installer.iss": [f'#define AppVersion "{__version__}"'],
            "packaging/build.ps1": [installer],
            "packaging/THIRD_PARTY_NOTICES.txt": [f"Z Calculator {__version__} - third-party notices"],
            "packaging/version.txt": [f"filevers=({numbers}), prodvers=({numbers})",
                                      f"'FileVersion', '{__version__}'", f"'ProductVersion', '{__version__}'"],
            "packaging/RELEASE_NOTES.md": [f"Z Calculator v{__version__}"],
            "README.md": [installer, f"### {__version__}"],
        }
        for name, fragments in expected.items():
            text = read(name)
            for fragment in fragments:
                with self.subTest(file=name, fragment=fragment):
                    self.assertIn(fragment, text)
        # No other release number may linger in the build inputs.
        for name in ("packaging/installer.iss", "packaging/build.ps1", "packaging/version.txt"):
            with self.subTest(file=name):
                self.assertEqual(set(re.findall(r"\d+\.\d+\.\d+(?![\d.])", read(name))) - {"6.22.3"}, {__version__})


if __name__ == "__main__":
    unittest.main()
