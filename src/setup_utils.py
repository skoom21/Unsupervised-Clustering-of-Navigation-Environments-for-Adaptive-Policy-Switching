from pathlib import Path
import shutil
import zipfile

import config

EXPECTED_COUNTS = {
    "maze": 60,
    "room": 40,
    "random": 70,
    "street": 90,
}

ZIP_MAP = {
    "maze": "maze-map.zip",
    "room": "room-map.zip",
    "random": "random-map.zip",
    "street": "street-map.zip",
}

DIR_MAP = {
    "maze": "maze-map",
    "room": "room-map",
    "random": "random-map",
    "street": "street-map",
}


def _count_maps(path: Path) -> int:
    return len(list(path.rglob("*.map")))


def extract_all_zips(zip_dir: Path, output_dir: Path) -> None:
    """Extract zip archives or copy pre-extracted folders into data/raw."""
    print("=== PHASE 0: Extracting map archives ===")
    if not zip_dir.exists():
        print(f"[WARNING] Zip directory does not exist: {zip_dir}")
        return

    for map_type, zip_name in ZIP_MAP.items():
        dest_dir = output_dir / map_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = zip_dir / zip_name
        src_dir = zip_dir / DIR_MAP[map_type]

        try:
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(dest_dir)
                print(f"[OK] Extracted {zip_name} -> {dest_dir}")
            elif src_dir.exists():
                map_files = list(src_dir.rglob("*.map"))
                if map_files:
                    for map_file in map_files:
                        shutil.copy2(map_file, dest_dir / map_file.name)
                    print(f"[OK] Copied {len(map_files)} maps from {src_dir} -> {dest_dir}")
                else:
                    print(f"[WARNING] No .map files in {src_dir}")
            else:
                print(f"[WARNING] Missing archive or folder for {map_type}")
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"[ERROR] Failed processing {map_type}: {exc}")

        count = _count_maps(dest_dir)
        expected = EXPECTED_COUNTS[map_type]
        if count != expected:
            print(f"[WARNING] {map_type} count {count} (expected {expected})")
        else:
            print(f"[OK] {map_type} count {count}")

    total = _count_maps(output_dir)
    if total != sum(EXPECTED_COUNTS.values()):
        print(f"[WARNING] Total maps {total} (expected {sum(EXPECTED_COUNTS.values())})")
    else:
        print(f"[OK] Total maps {total}")
