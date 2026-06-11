"""
setup_datalake.py

Creates the DataLake folder structure on Google Drive and uploads
the local dataset. Run once from the project root:

    python setup_datalake.py
    python setup_datalake.py --dry-run   # preview without uploading
"""

import argparse
from pathlib import Path

from utils.auth import get_credentials
from utils.drive import get_or_create_folder, upload_file


# ── Config ────────────────────────────────────────────────────────────────────

LOCAL_DATA = Path("data")       # local folder containing Train/ and Test/
LAKE_ROOT  = "DataLake"         # top-level folder name in Google Drive

# Folders to create under DataLake (mirrors the structure we designed)
LAKE_STRUCTURE = [
    "01_raw/Train/Overripe",
    "01_raw/Train/Ripe",
    "01_raw/Train/Unripe",
    "01_raw/Test/Overripe",
    "01_raw/Test/Ripe",
    "01_raw/Test/Unripe",
    "02_processed/Train",
    "02_processed/Val",
    "02_processed/Test",
    "03_artifacts/models",
    "03_artifacts/quantized",
    "03_artifacts/metrics",
]

# Map local folder → Drive destination path
UPLOAD_MAPPING = {
    "Train/Overripe" : "01_raw/Train/Overripe",
    "Train/Ripe"     : "01_raw/Train/Ripe",
    "Train/Unripe"   : "01_raw/Train/Unripe",
    "Test/Overripe"  : "01_raw/Test/Overripe",
    "Test/Ripe"      : "01_raw/Test/Ripe",
    "Test/Unripe"    : "01_raw/Test/Unripe",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_folder_tree(root_id: str, svc) -> dict[str, str]:
    """Create all folders in LAKE_STRUCTURE, return {path: folder_id} map."""
    folder_ids = {"": root_id}

    for full_path in LAKE_STRUCTURE:
        parts = Path(full_path).parts
        current_parent = root_id

        for i, part in enumerate(parts):
            partial = str(Path(*parts[: i + 1]))
            if partial not in folder_ids:
                print(f"  [folder] creating {LAKE_ROOT}/{partial}")
                fid = get_or_create_folder(part, parent_id=current_parent, service=svc)
                folder_ids[partial] = fid
            current_parent = folder_ids[partial]

    return folder_ids


def upload_dataset(folder_ids: dict[str, str], svc, dry_run: bool = False) -> None:
    """Upload every file from LOCAL_DATA into the matching Drive folder."""
    total, skipped = 0, 0

    for local_rel, drive_path in UPLOAD_MAPPING.items():
        local_folder = LOCAL_DATA / local_rel

        if not local_folder.exists():
            print(f"  [skip] local folder not found: {local_folder}")
            skipped += 1
            continue

        files = [f for f in local_folder.rglob("*") if f.is_file()]

        if not files:
            print(f"  [skip] no files in {local_folder}")
            skipped += 1
            continue

        folder_id = folder_ids[drive_path]

        for file in files:
            if dry_run:
                print(f"  [dry-run] {file}  →  {LAKE_ROOT}/{drive_path}/{file.name}")
            else:
                result = upload_file(str(file), folder_id=folder_id, service=svc)
                print(f"  [upload] {file.name}  →  {LAKE_ROOT}/{drive_path}  (id: {result['id']})")
            total += 1

    print(f"\n{'[dry-run] ' if dry_run else ''}Done — {total} files, {skipped} folders skipped.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    """Build the DataLake structure and upload the local dataset."""
    print(f"\n{'=' * 52}")
    print(f"  DataLake setup {'(dry-run)' if dry_run else ''}")
    print(f"  Local data : {LOCAL_DATA.resolve()}")
    print(f"  Drive root : {LAKE_ROOT}")
    print(f"{'=' * 52}\n")

    if not LOCAL_DATA.exists():
        raise FileNotFoundError(f"Local data folder not found: {LOCAL_DATA.resolve()}")

    print("[1/3] Authenticating with Google Drive...")
    creds = get_credentials()

    from googleapiclient.discovery import build
    svc = build("drive", "v3", credentials=creds)

    print("\n[2/3] Building folder structure...")
    root_id = get_or_create_folder(LAKE_ROOT, service=svc)
    print(f"  [folder] {LAKE_ROOT}  (id: {root_id})")
    folder_ids = build_folder_tree(root_id, svc)

    print(f"\n[3/3] Uploading dataset{' (dry-run)' if dry_run else ''}...")
    upload_dataset(folder_ids, svc, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up DataLake on Google Drive.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be uploaded without actually uploading.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
