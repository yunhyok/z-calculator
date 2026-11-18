"""Local Cadence CMX dielectric tables; no vendor data is bundled.

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


def load_cmx(path: str | Path) -> dict[str, Material]:
    """Read Cadence Material Library XML and select its default dielectric model.

    Untagged models are the library default (temperature unspecified). When
    only tagged models exist, use the stratum nearest 25 C, lower on a tie.
    Empty dielectric records and materials without dielectric data are skipped.
    Invalid populated records fail the import, avoiding partially applied data.
    """
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("CMX 파일이 가져오기 한도(16 MiB)를 넘습니다.")
    data = path.read_bytes()
    # DTDs and entities are unnecessary for CMX numeric tables.
    normalized = data.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in normalized or b"<!ENTITY" in normalized:
        raise ValueError("DTD나 엔티티가 포함된 CMX 파일은 지원하지 않습니다.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"CMX XML을 읽을 수 없습니다: {exc}") from exc
    if root.tag != "Cadence_Material_Lib":
        raise ValueError("Cadence_Material_Lib 형식의 CMX 문서가 아닙니다.")
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
    if not materials:
        raise ValueError("CMX 파일에 유전체 모델 데이터가 없습니다.")
    return materials
