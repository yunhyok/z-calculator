param([string]$InnoCompiler = "")

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (!(Test-Path -LiteralPath $python)) { throw "Create the project .venv and install requirements.txt first." }
if (!$InnoCompiler) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 7\ISCC.exe"
    )
    $InnoCompiler = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (!$InnoCompiler -or !(Test-Path -LiteralPath $InnoCompiler)) { throw "Install Inno Setup 6, or pass -InnoCompiler <ISCC.exe>." }

Push-Location $projectRoot
$originalPath = $env:PATH
try {
    foreach ($required in @("main.py", "assets\zcalc.ico", "requirements.txt")) {
        if (!(Test-Path -LiteralPath $required)) { throw "Missing required build input: $required" }
    }
    $env:PYTHONHASHSEED = "0"
    $pythonBase = (& $python -c "import sys; print(sys.base_prefix)").Trim()
    # Host tools can put incompatible ICU/CRT DLLs on PATH before Windows DLLs.
    $env:PATH = @((Split-Path $python -Parent), $pythonBase, "$pythonBase\DLLs", "$env:SystemRoot\System32", $env:SystemRoot) -join ";"
    & $python -m pip check
    if ($LASTEXITCODE) { throw "Installed dependency checks failed." }
    & $python -m pytest -q
    if ($LASTEXITCODE) { throw "Tests failed; no installer was built." }

    @'
from importlib.metadata import distribution
from pathlib import Path
import json
import shutil
import sys

root = Path.cwd()
assert sys.maxsize > 2**32, "Build with 64-bit Python."
assert distribution("pyinstaller").version == "6.22.3", "Install pyinstaller==6.22.3."
for requirement in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
    if requirement.strip() and not requirement.startswith("#"):
        name, version = requirement.strip().split("==")
        assert distribution(name).version == version, f"Install the pinned {requirement}."

licenses = root / "build" / "licenses"
licenses.mkdir(parents=True, exist_ok=True)
packages = ["numpy", "scipy", "PySide6-Essentials", "shiboken6", "python-pptx", "lxml",
            "Pillow", "XlsxWriter", "typing_extensions", "pyinstaller"]
versions = {"Python": sys.version.split()[0]}
for name in packages:
    package = distribution(name)
    versions[name] = package.version
    for item in package.files or []:
        if any(token in str(item).lower() for token in ("license", "copying", "notice")) or item.name == "METADATA":
            source = Path(package.locate_file(item))
            if source.is_file():
                destination = licenses / name / Path(*[part for part in item.parts if part != ".."])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
shutil.copy2(Path(sys.base_prefix) / "LICENSE.txt", licenses / "PYTHON-LICENSE.txt")
for source in (root / "packaging" / "licenses").iterdir():
    shutil.copy2(source, licenses / source.name)
shutil.copy2(root / "packaging" / "THIRD_PARTY_NOTICES.txt", licenses / "THIRD_PARTY_NOTICES.txt")
(licenses / "package-versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
print("Build dependencies:", json.dumps(versions))
'@ | & $python -
    if ($LASTEXITCODE) { throw "Pinned dependency or license collection checks failed." }

    & $python -m PyInstaller --noconfirm --clean --onedir --windowed --name ZCalculator `
        --icon "$projectRoot\assets\zcalc.ico" --add-data "$projectRoot\assets\zcalc.ico;assets" `
        --add-data "$projectRoot\build\licenses;licenses" --version-file "$projectRoot\packaging\version.txt" `
        --collect-data pptx --exclude-module pytest --exclude-module _pytest `
        --distpath dist --workpath "build\pyinstaller" --specpath build main.py
    if ($LASTEXITCODE) { throw "PyInstaller failed." }
    if (Get-ChildItem -LiteralPath "dist\ZCalculator" -Recurse -Filter "*.cmx") {
        throw "Unexpected private material library in frozen output."
    }
    @'
import ast
import os
from pathlib import Path
import sys
from PyInstaller.archive.readers import ZlibArchiveReader

root = Path.cwd()
allowed = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), Path(os.environ["SystemRoot"]).resolve()]
table = ast.literal_eval((root / "build/pyinstaller/ZCalculator/Analysis-00.toc").read_text(encoding="utf-8"))
for group in table:
    if isinstance(group, list):
        for item in group:
            if isinstance(item, tuple) and len(item) == 3 and item[2] == "BINARY":
                source = Path(item[1]).resolve()
                assert any(source.is_relative_to(base) for base in allowed), f"Foreign native library: {source}"
archive = ZlibArchiveReader(str(root / "build/pyinstaller/ZCalculator/PYZ-00.pyz"))
for name in ["zcalc.visuals", "zcalc.ui", "zcalc.engine", "zcalc.materials", "zcalc.touchstone"]:
    code = archive.extract(name)
    source = root / (name.replace(".", "/") + ".py")
    assert code == compile(source.read_text(encoding="utf-8"), code.co_filename, "exec"), f"Stale frozen source: {name}"
assert not (root / "dist/ZCalculator/_internal/icuuc.dll").exists(), "Qt must use the Windows system ICU."
print("Frozen source and native-library provenance checks passed.")
'@ | & $python -
    if ($LASTEXITCODE) { throw "Frozen source or native-library provenance checks failed." }

    & $InnoCompiler /Qp "packaging\installer.iss"
    if ($LASTEXITCODE) { throw "Inno Setup failed." }
    $installer = Join-Path $projectRoot "dist\Z-Calculator-1.2.1-Windows-x64-Setup.exe"
    Get-FileHash -Algorithm SHA256 -LiteralPath $installer
    Write-Host "Installer: $installer"
}
finally {
    $env:PATH = $originalPath
    Pop-Location
}
