"""Local Cadence CMX material tables; no vendor data is bundled.

ValueError messages are user-facing Korean text.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True, eq=False)
class Material:
    name: str
    frequencies_hz: np.ndarray
    er: np.ndarray
    df: np.ndarray
    temperature_c: float | None = None
    source: str = "manual"

    def __post_init__(self):
        for field, label in (("frequencies_hz", "주파수"), ("er", "Dk"), ("df", "Df")):
            values = np.array(getattr(self, field), dtype=float, copy=True)
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"{self.name}: {label}에 유한한 숫자가 필요합니다.")
            values.setflags(write=False)
            object.__setattr__(self, field, values)
        if len(self.er) != len(self.df) or len(self.er) != len(self.frequencies_hz):
            raise ValueError(f"{self.name}: 주파수·Dk·Df 표의 행 수가 다릅니다.")
        if np.any(self.frequencies_hz <= 0) or np.any(np.diff(self.frequencies_hz) <= 0):
            raise ValueError(f"{self.name}: 주파수는 0보다 크고 오름차순이어야 합니다.")
        if np.any(self.er < 1) or np.any(self.df < 0):
            raise ValueError(f"{self.name}: Dk는 1 이상, Df는 0 이상이어야 합니다.")
        if self.temperature_c is not None and not np.isfinite(self.temperature_c):
            raise ValueError(f"{self.name}: 온도는 유한한 값이어야 합니다.")

    @property
    def frequency_range_hz(self) -> tuple[float, float]:
        """Measured/reference table range, including a single reference point."""
        return float(self.frequencies_hz[0]), float(self.frequencies_hz[-1])

    @property
    def is_constant(self) -> bool:
        return len(self.frequencies_hz) == 1

    def sample(self, freq_hz) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate on log frequency; hold endpoints outside the table range.

        A one-row table is a constant assumption at every frequency, not a
        measurement establishing broadband behavior.
        """
        frequency = np.asarray(freq_hz, dtype=float)
        if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
            raise ValueError("주파수는 0보다 큰 유한한 값이어야 합니다.")
        position = np.log10(frequency)
        table_position = np.log10(self.frequencies_hz)
        return tuple(
            np.asarray(np.interp(position, table_position, values))
            for values in (self.er, self.df)
        )


def manual_material(er: float, df: float) -> Material:
    """Create a frequency-independent material from manually entered Dk and Df."""
    return Material("직접 입력", np.array([1e9]), np.array([er]), np.array([df]))


@dataclass(frozen=True)
class Conductor:
    name: str
    conductivity_s_m: float
    temperature_c: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.conductivity_s_m) or self.conductivity_s_m <= 0:
            raise ValueError(f"{self.name}: 전도율은 0보다 큰 유한한 값이어야 합니다.")
        if self.temperature_c is not None and not np.isfinite(self.temperature_c):
            raise ValueError(f"{self.name}: 온도는 유한한 값이어야 합니다.")


def _read_cmx(path: str | Path) -> tuple[Path, ET.Element]:
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("CMX 파일이 가져오기 한도(16 MiB)를 넘습니다.")
    data = path.read_bytes()
    # DTDs and entities are unnecessary for CMX numeric tables.
    normalized = data.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in normalized or b"<!ENTITY" in normalized:
        raise ValueError("DTD나 엔티티가 포함된 CMX 파일은 지원하지 않습니다.")
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            data = data.decode("utf-16")
        elif data.startswith(b"<\x00"):
            data = data.decode("utf-16-le")
        elif data.startswith(b"\x00<"):
            data = data.decode("utf-16-be")
        root = ET.fromstring(data)
    except (ET.ParseError, UnicodeError, LookupError, ValueError) as exc:
        raise ValueError(f"CMX XML을 읽을 수 없습니다: {exc}") from exc
    if root.tag != "Cadence_Material_Lib":
        raise ValueError("Cadence_Material_Lib 형식의 CMX 문서가 아닙니다.")
    return path, root


def _load_dielectrics(path: Path, root: ET.Element) -> dict[str, Material]:
    description = root.find("./DataDescriptions/Dielectric")
    columns = list(description) if description is not None else []
    names = [column.get("name", "") for column in columns]
    if names and names != ["Temperature", "Frequency", "Relative Permittivity", "LossTangent"]:
        raise ValueError("지원하지 않는 CMX 유전체 열 구성입니다.")
    unit = columns[1].get("unit", "MHz") if columns else "MHz"
    scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(unit.lower())
    if scale is None:
        raise ValueError(f"지원하지 않는 CMX 주파수 단위입니다: {unit}")
    if columns and columns[0].get("unit", "C") not in ("C", "degC", "°C"):
        raise ValueError("CMX 온도 단위는 섭씨(°C)만 지원합니다.")

    materials = {}
    for node in root.findall("Material"):
        models = [model for model in node.findall("./Dielectric/Model") if (model.text or "").strip()]
        if not models:
            continue
        name = (node.get("name") or "").strip()
        if not name or name in materials:
            raise ValueError(f"CMX 유전체 재료 이름이 없거나 중복됩니다: {name!r}")
        try:
            temperature_models = [
                (float(model.get("Temperature", "25")), model) for model in models
            ]
            if any(not np.isfinite(temp) for temp, _ in temperature_models):
                raise ValueError("온도 값이 유한하지 않습니다")
            model = next((m for m in models if "Temperature" not in m.attrib), None)
            if model is None:
                _, model = min(temperature_models, key=lambda pair: (abs(pair[0] - 25), pair[0]))
            temperature = float(model.attrib["Temperature"]) if "Temperature" in model.attrib else None
            rows = []
            for line in (model.text or "").splitlines():
                if line.strip():
                    values = [float(value) for value in line.split()]
                    if len(values) != 3:
                        raise ValueError("각 행은 주파수 Dk Df 세 값이어야 합니다")
                    rows.append(values)
            table = np.array(sorted(rows), dtype=float)
            unique_rows = []
            for row in table:
                if unique_rows and row[0] == unique_rows[-1][0]:
                    if not np.array_equal(row, unique_rows[-1]):
                        raise ValueError("같은 주파수에 서로 다른 값이 있습니다")
                else:
                    unique_rows.append(row)
            table = np.array(unique_rows)
            material = Material(name, table[:, 0] * scale, table[:, 1], table[:, 2], temperature, str(path))
        except (TypeError, ValueError, OverflowError) as exc:
            reason = str(exc).removeprefix(f"{name}: ")
            raise ValueError(f"{name}: CMX 유전체 모델이 올바르지 않습니다 ({reason})") from exc
        materials[name] = material
    return materials


def _load_conductors(root: ET.Element) -> dict[str, Conductor]:
    description = root.find("./DataDescriptions/Metal")
    columns = list(description) if description is not None else []
    if columns and [column.get("name", "") for column in columns] != ["Temperature", "Conductivity"]:
        raise ValueError("지원하지 않는 CMX 도체 열 구성입니다.")
    if columns and columns[0].get("unit", "C") not in ("C", "degC", "°C"):
        raise ValueError("CMX 도체 온도 단위는 섭씨(°C)만 지원합니다.")
    unit = columns[1].get("unit", "S/m").strip() if columns else "S/m"
    scale = {"S/m": 1.0, "S/cm": 100.0, "S/mm": 1000.0, "MS/m": 1e6, "mS/m": 1e-3}.get(unit)
    if scale is None:
        raise ValueError(f"지원하지 않는 CMX 전도율 단위입니다: {unit}")
    conductors = {}
    for node in root.findall("Material"):
        models = [model for model in node.findall("./Metal/Model") if (model.text or "").strip()]
        if not models:
            continue
        name = (node.get("name") or "").strip()
        if not name or name in conductors:
            raise ValueError(f"CMX 도체 재료 이름이 없거나 중복됩니다: {name!r}")
        try:
            rows = []
            constants = []
            for model in models:
                temperature = float(model.attrib["Temperature"]) if "Temperature" in model.attrib else None
                if temperature is not None and not np.isfinite(temperature):
                    raise ValueError("온도 값이 유한하지 않습니다")
                for line in (model.text or "").splitlines():
                    if not line.strip():
                        continue
                    values = [float(value) for value in line.split()]
                    if not np.all(np.isfinite(values)):
                        raise ValueError("온도와 전도율은 유한한 값이어야 합니다")
                    if len(values) == 2:
                        rows.append(values)
                    elif len(values) == 1:
                        if temperature is None:
                            constants.append(values[0])
                        else:
                            rows.append([temperature, values[0]])
                    else:
                        raise ValueError("각 행은 온도·전도율 두 값 또는 상수 전도율 한 값이어야 합니다")
            if constants:
                if rows or any(value != constants[0] for value in constants):
                    raise ValueError("온도 미지정 상수와 온도별 전도율을 함께 지정할 수 없습니다")
                conductor = Conductor(name, constants[0] * scale)
            else:
                table = np.array(sorted(rows), dtype=float)
                if np.any(table[:, 1] <= 0):
                    raise ValueError("전도율은 0보다 커야 합니다")
                unique_rows = []
                for row in table:
                    if unique_rows and row[0] == unique_rows[-1][0]:
                        if row[1] != unique_rows[-1][1]:
                            raise ValueError("같은 온도에 서로 다른 전도율이 있습니다")
                    else:
                        unique_rows.append(row)
                table = np.array(unique_rows)
                actual_temperature = float(np.clip(20.0, table[0, 0], table[-1, 0]))
                conductivity = float(np.interp(actual_temperature, table[:, 0], table[:, 1])) * scale
                conductor = Conductor(name, conductivity, actual_temperature)
        except (TypeError, ValueError, OverflowError) as exc:
            reason = str(exc).removeprefix(f"{name}: ")
            raise ValueError(f"{name}: CMX 도체 모델이 올바르지 않습니다 ({reason})") from exc
        conductors[name] = conductor
    return conductors


def load_cmx(path: str | Path) -> dict[str, Material]:
    """Load dielectric models only, preserving the original public interface.

    Prefer an untagged default; otherwise choose the stratum nearest 25 C.
    Metal data is ignored, including records unsupported by the conductor UI.
    """
    path, root = _read_cmx(path)
    materials = _load_dielectrics(path, root)
    if not materials:
        raise ValueError("CMX 파일에 유전체 모델 데이터가 없습니다.")
    return materials


def load_material_library(path: str | Path) -> tuple[dict[str, Material], dict[str, Conductor]]:
    """Read a mixed, dielectric-only or metal-only CMX document exactly once.

    Metal conductivity is interpolated linearly at 20 C, with endpoint holds.
    temperature_c records the actual interpolation/clamp temperature; None is
    reserved for an untagged, single-value constant with unspecified temperature.
    Values such as a finite PEC conductivity are retained exactly as supplied.
    """
    path, root = _read_cmx(path)
    materials = _load_dielectrics(path, root)
    conductors = _load_conductors(root)
    if not materials and not conductors:
        raise ValueError("CMX 파일에 유전체나 도체 모델 데이터가 없습니다.")
    return materials, conductors
