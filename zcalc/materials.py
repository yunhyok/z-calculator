"""Local Cadence CMX dielectric tables; no vendor data is bundled."""

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
        for field in ("frequencies_hz", "er", "df"):
            values = np.array(getattr(self, field), dtype=float, copy=True)
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"{self.name}: {field} must contain finite numbers.")
            values.setflags(write=False)
            object.__setattr__(self, field, values)
        if len(self.er) != len(self.df) or len(self.er) != len(self.frequencies_hz):
            raise ValueError(f"{self.name}: frequency, Dk and Df table sizes differ.")
        if np.any(self.frequencies_hz <= 0) or np.any(np.diff(self.frequencies_hz) <= 0):
            raise ValueError(f"{self.name}: frequencies must be positive and increasing.")
        if np.any(self.er < 1) or np.any(self.df < 0):
            raise ValueError(f"{self.name}: Dk must be at least 1 and Df nonnegative.")
        if self.temperature_c is not None and not np.isfinite(self.temperature_c):
            raise ValueError(f"{self.name}: temperature must be finite.")

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
            raise ValueError("Frequency must be finite and greater than zero.")
        position = np.log10(frequency)
        table_position = np.log10(self.frequencies_hz)
        return tuple(
            np.asarray(np.interp(position, table_position, values))
            for values in (self.er, self.df)
        )


def manual_material(er: float, df: float) -> Material:
    """Create a frequency-independent material from manually entered Dk and Df."""
    return Material("Manual", np.array([1e9]), np.array([er]), np.array([df]))


def load_cmx(path: str | Path) -> dict[str, Material]:
    """Read Cadence Material Library XML and select its default dielectric model.

    Untagged models are the library default (temperature unspecified). When
    only tagged models exist, use the stratum nearest 25 C, lower on a tie.
    Empty dielectric records and materials without dielectric data are skipped.
    Invalid populated records fail the import, avoiding partially applied data.
    """
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("CMX file exceeds the 16 MiB import limit.")
    data = path.read_bytes()
    # DTDs and entities are unnecessary for CMX numeric tables.
    normalized = data.replace(b"\x00", b"").upper()
    if b"<!DOCTYPE" in normalized or b"<!ENTITY" in normalized:
        raise ValueError("CMX files containing DTDs or entities are not supported.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid CMX XML: {exc}") from exc
    if root.tag != "Cadence_Material_Lib":
        raise ValueError("Expected a Cadence_Material_Lib CMX document.")
    description = root.find("./DataDescriptions/Dielectric")
    columns = list(description) if description is not None else []
    names = [column.get("name", "") for column in columns]
    if names and names != ["Temperature", "Frequency", "Relative Permittivity", "LossTangent"]:
        raise ValueError("Unsupported CMX dielectric column order.")
    unit = columns[1].get("unit", "MHz") if columns else "MHz"
    scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(unit.lower())
    if scale is None:
        raise ValueError(f"Unsupported CMX frequency unit: {unit}")
    if columns and columns[0].get("unit", "C") not in ("C", "degC", "°C"):
        raise ValueError("Only Celsius CMX temperature strata are supported.")

    materials = {}
    for node in root.findall("Material"):
        models = [model for model in node.findall("./Dielectric/Model") if (model.text or "").strip()]
        if not models:
            continue
        name = (node.get("name") or "").strip()
        if not name or name in materials:
            raise ValueError(f"Missing or duplicate CMX dielectric material name: {name!r}")
        try:
            temperature_models = [
                (float(model.get("Temperature", "25")), model) for model in models
            ]
            if any(not np.isfinite(temp) for temp, _ in temperature_models):
                raise ValueError("nonfinite temperature")
            model = next((m for m in models if "Temperature" not in m.attrib), None)
            if model is None:
                _, model = min(temperature_models, key=lambda pair: (abs(pair[0] - 25), pair[0]))
            temperature = float(model.attrib["Temperature"]) if "Temperature" in model.attrib else None
            rows = []
            for line in (model.text or "").splitlines():
                if line.strip():
                    values = [float(value) for value in line.split()]
                    if len(values) != 3:
                        raise ValueError("expected Frequency Dk Df rows")
                    rows.append(values)
            table = np.array(sorted(rows), dtype=float)
            unique_rows = []
            for row in table:
                if unique_rows and row[0] == unique_rows[-1][0]:
                    if not np.array_equal(row, unique_rows[-1]):
                        raise ValueError("conflicting rows at the same frequency")
                else:
                    unique_rows.append(row)
            table = np.array(unique_rows)
            material = Material(name, table[:, 0] * scale, table[:, 1], table[:, 2], temperature, str(path))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}: invalid CMX dielectric model ({exc})") from exc
        materials[name] = material
    if not materials:
        raise ValueError("No populated dielectric models found in the CMX file.")
    return materials
