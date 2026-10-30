"""Dataset discovery: pair every XML with its source and reference TIFF.

Nothing about the six official scenes is hard-coded. A scene is defined by an
XML file; its stem is the scene ID, and the rasters are the TIFFs named
``<scene_id>_source*.tif`` and ``<scene_id>_reference*.tif``.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

TIFF_SUFFIXES = {".tif", ".tiff"}

# Chandrayaan-2 product IDs start with a three-letter instrument code.
SENSOR_PREFIXES = {"OHR": "OHRC", "IIR": "IIRS", "TMC": "TMC-2", "TM2": "TMC-2"}


class SceneRecord(BaseModel):
    scene_id: str
    sensor: str
    xml_path: str
    source_path: str
    reference_path: str
    reference_sensor: str | None = None


class IncompleteScene(BaseModel):
    scene_id: str
    sensor: str
    xml_path: str
    missing: list[str]


class DatasetIndex(BaseModel):
    data_root: str
    exists: bool
    scenes: list[SceneRecord]
    incomplete: list[IncompleteScene]
    unpaired_rasters: list[str]
    sensor_counts: dict[str, int]

    def get(self, scene_id: str) -> SceneRecord | None:
        return next((s for s in self.scenes if s.scene_id == scene_id), None)

    def by_sensor(self, sensor: str) -> list[SceneRecord]:
        return [s for s in self.scenes if s.sensor == sensor]


def sensor_from_scene_id(scene_id: str) -> str:
    return SENSOR_PREFIXES.get(scene_id[:3].upper(), "UNKNOWN")


def reference_sensor_for(sensor: str) -> str | None:
    """Reference imagery supplied by the mentor for each Chandrayaan-2 sensor."""
    return {"OHRC": "LRO-NAC", "IIRS": "LRO-WAC"}.get(sensor)


def _role_of(tif: Path, scene_id: str) -> str | None:
    """'source' / 'reference' if ``tif`` belongs to ``scene_id``, else None.

    The character right after the ID must start the role word, so scene
    ``..._V2_1`` never claims the rasters of ``..._V2_1_01``.
    """
    name = tif.name
    if not name.lower().startswith(scene_id.lower() + "_"):
        return None
    rest = name[len(scene_id) + 1 :].lower()
    for role in ("source", "reference"):
        if rest.startswith(role):
            return role
    return None


def discover_scenes(data_root: str | Path) -> DatasetIndex:
    root = Path(data_root)
    if not root.is_dir():
        return DatasetIndex(
            data_root=str(root), exists=False, scenes=[], incomplete=[], unpaired_rasters=[], sensor_counts={}
        )

    xmls = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")
    tifs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)

    scenes: list[SceneRecord] = []
    incomplete: list[IncompleteScene] = []
    claimed: set[Path] = set()

    for xml in xmls:
        scene_id = xml.stem
        sensor = sensor_from_scene_id(scene_id)
        found: dict[str, Path] = {}
        # Prefer rasters sitting next to the XML, then anywhere under the root.
        for tif in sorted(tifs, key=lambda t: (t.parent != xml.parent, str(t))):
            role = _role_of(tif, scene_id)
            if role and role not in found:
                found[role] = tif
        missing = [r for r in ("source", "reference") if r not in found]
        if missing:
            incomplete.append(
                IncompleteScene(scene_id=scene_id, sensor=sensor, xml_path=str(xml), missing=missing)
            )
            continue
        claimed.update(found.values())
        scenes.append(
            SceneRecord(
                scene_id=scene_id,
                sensor=sensor,
                xml_path=str(xml),
                source_path=str(found["source"]),
                reference_path=str(found["reference"]),
                reference_sensor=reference_sensor_for(sensor),
            )
        )

    scenes.sort(key=lambda s: (s.sensor != "OHRC", s.sensor, s.scene_id))
    counts: dict[str, int] = {}
    for s in scenes:
        counts[s.sensor] = counts.get(s.sensor, 0) + 1

    return DatasetIndex(
        data_root=str(root),
        exists=True,
        scenes=scenes,
        incomplete=incomplete,
        unpaired_rasters=[str(t) for t in tifs if t not in claimed],
        sensor_counts=counts,
    )
