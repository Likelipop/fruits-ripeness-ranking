import mimetypes
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from utils.auth import get_credentials


def _build_service(credentials=None):
    creds = credentials or get_credentials()
    return build("drive", "v3", credentials=creds)


def get_or_create_folder(folder_name: str, parent_id: str = None, service=None) -> str:
    """Return the ID of a Drive folder, creating it if it doesn't exist.

    Args:
        folder_name: Name of the folder.
        parent_id: Parent folder ID. Defaults to Drive root.
        service: Authenticated Drive service. Created automatically if omitted.

    Returns:
        Folder ID string.
    """
    svc = service or _build_service()
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = svc.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        **({"parents": [parent_id]} if parent_id else {}),
    }
    folder = svc.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def upload_file(
    local_path: str,
    folder_id: str = None,
    destination_name: str = None,
    resumable: bool = True,
    service=None,
) -> dict:
    """Upload a local file to Google Drive.

    Args:
        local_path: Absolute or relative path to the file to upload.
        folder_id: Target Drive folder ID. Uploads to root if omitted.
        destination_name: Override the filename in Drive. Defaults to the local filename.
        resumable: Use resumable upload (recommended for files > 5 MB).
        service: Authenticated Drive service. Created automatically if omitted.

    Returns:
        Dict with ``id`` and ``name`` of the uploaded file.

    Raises:
        FileNotFoundError: If ``local_path`` does not exist.
        HttpError: On Drive API errors.
    """
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    svc = service or _build_service()
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "application/octet-stream"

    metadata = {
        "name": destination_name or path.name,
        **({"parents": [folder_id]} if folder_id else {}),
    }

    media = MediaFileUpload(str(path), mimetype=mime_type, resumable=resumable)

    try:
        file = (
            svc.files()
            .create(body=metadata, media_body=media, fields="id, name")
            .execute()
        )
        return {"id": file["id"], "name": file["name"]}
    except HttpError as error:
        raise HttpError(error.resp, error.content) from error


def upload_to_data_lake(
    local_path: str,
    lake_root: str = "DataLake",
    sub_folder: str = None,
    service=None,
) -> dict:
    """Upload a file into a structured DataLake folder hierarchy on Drive.

    Folder structure: ``<lake_root>/<sub_folder (optional)>/<file>``

    Args:
        local_path: Path to the local file.
        lake_root: Top-level Drive folder name acting as the data lake root.
        sub_folder: Optional sub-folder inside the lake (e.g. ``"raw/2024-06"``).
        service: Authenticated Drive service. Created automatically if omitted.

    Returns:
        Dict with ``id``, ``name``, and ``folder_id`` of the uploaded file.
    """
    svc = service or _build_service()

    folder_id = get_or_create_folder(lake_root, service=svc)

    if sub_folder:
        for part in Path(sub_folder).parts:
            folder_id = get_or_create_folder(part, parent_id=folder_id, service=svc)

    result = upload_file(local_path, folder_id=folder_id, service=svc)
    result["folder_id"] = folder_id
    return result
