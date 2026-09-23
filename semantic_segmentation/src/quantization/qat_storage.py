from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

from src.quantization.core import save_json

def _secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret(name)
        except Exception:
            value = None
    if not value:
        raise RuntimeError(f"Secret unavailable: {name}")
    return value


def build_drive_service():
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Google Drive support requires quantization dependencies") from exc
    credentials = Credentials(
        token=None,
        refresh_token=_secret("GDRIVE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=_secret("GDRIVE_CLIENT_ID"),
        client_secret=_secret("GDRIVE_CLIENT_SECRET"),
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _drive_escape(value) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def resolve_drive_folder(service, cfg) -> str:
    clauses = [
        f"name = '{_drive_escape(cfg.output.drive_folder_name)}'",
        "mimeType = 'application/vnd.google-apps.folder'",
        "trashed = false",
    ]
    if cfg.output.drive_parent_folder_id:
        clauses.append(
            f"'{_drive_escape(cfg.output.drive_parent_folder_id)}' in parents"
        )
    files = service.files().list(
        q=" and ".join(clauses), fields="files(id,name)", pageSize=10
    ).execute().get("files", [])
    if len(files) > 1:
        raise RuntimeError(f"Multiple Drive folders named {cfg.output.drive_folder_name}")
    if files:
        return files[0]["id"]
    metadata = {
        "name": str(cfg.output.drive_folder_name),
        "mimeType": "application/vnd.google-apps.folder",
    }
    if cfg.output.drive_parent_folder_id:
        metadata["parents"] = [str(cfg.output.drive_parent_folder_id)]
    return service.files().create(body=metadata, fields="id").execute()["id"]


def upload_file(path: Path, cfg, service=None, folder_id=None):
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise RuntimeError("Google Drive support requires quantization dependencies") from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    service = service or build_drive_service()
    folder_id = folder_id or resolve_drive_folder(service, cfg)
    files = service.files().list(
        q=(
            f"'{_drive_escape(folder_id)}' in parents and "
            f"name = '{_drive_escape(path.name)}' and trashed = false"
        ),
        fields="files(id,name)",
        pageSize=10,
    ).execute().get("files", [])
    if len(files) > 1:
        raise RuntimeError(f"Multiple Drive files named {path.name}")
    media = MediaFileUpload(str(path), resumable=True)
    if files:
        return service.files().update(
            fileId=files[0]["id"], media_body=media, fields="id,name"
        ).execute()
    return service.files().create(
        body={"name": path.name, "parents": [folder_id]},
        media_body=media,
        fields="id,name",
    ).execute()


def download_output_artifact(filename: str, destination: Path, cfg) -> Path:
    if destination.is_file():
        return destination
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError("Google Drive support requires quantization dependencies") from exc
    service = build_drive_service()
    folder_id = resolve_drive_folder(service, cfg)
    files = service.files().list(
        q=(
            f"'{_drive_escape(folder_id)}' in parents and "
            f"name = '{_drive_escape(filename)}' and trashed = false"
        ),
        fields="files(id,name)",
        pageSize=10,
    ).execute().get("files", [])
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one {filename} in {cfg.output.drive_folder_name}, found {len(files)}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=files[0]["id"])
    with destination.open("wb") as stream:
        downloader = MediaIoBaseDownload(stream, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return destination


def save_results(results, cfg, paths):
    results["configuration"] = OmegaConf.to_container(cfg, resolve=True)
    save_json(results, paths.results)
