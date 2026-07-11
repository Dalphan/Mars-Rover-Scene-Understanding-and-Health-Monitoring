from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from src.utils.artifacts import sha256_file


DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
LOGGER = logging.getLogger(__name__)


def get_kaggle_secret(label: str) -> str:
    """Read one secret without ever logging its value."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as error:
        raise RuntimeError("kaggle_secrets is available only in Kaggle notebooks") from error
    value = UserSecretsClient().get_secret(label)
    if not value:
        raise RuntimeError(f"Missing required Kaggle secret: {label}")
    return value


def build_drive_service_from_kaggle_secrets():
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as error:
        raise RuntimeError(
            "Google Drive support requires google-auth and google-api-python-client"
        ) from error

    credentials = Credentials(
        token=None,
        refresh_token=get_kaggle_secret("GDRIVE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=get_kaggle_secret("GDRIVE_CLIENT_ID"),
        client_secret=get_kaggle_secret("GDRIVE_CLIENT_SECRET"),
        scopes=[DRIVE_FILE_SCOPE],
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _escape_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


class GoogleDriveRunStore:
    """Minimal idempotent Drive v3 store for run folders and artifacts."""

    folder_mime_type = "application/vnd.google-apps.folder"

    def __init__(self, service, parent_folder_id: str, logger=None) -> None:
        if not parent_folder_id:
            raise ValueError("parent_folder_id must not be empty")
        self.service = service
        self.parent_folder_id = parent_folder_id
        self.logger = logger or LOGGER

    def _list_children(self, parent_id: str, name: str | None = None) -> list[dict[str, Any]]:
        query = [f"'{_escape_query_value(parent_id)}' in parents", "trashed = false"]
        if name is not None:
            query.append(f"name = '{_escape_query_value(name)}'")
        response = (
            self.service.files()
            .list(
                q=" and ".join(query),
                spaces="drive",
                fields="files(id,name,mimeType,size,md5Checksum)",
                pageSize=1000,
            )
            .execute(num_retries=3)
        )
        return response.get("files", [])

    def create_run_folder(self, run_id: str, *, parent_id: str | None = None) -> str:
        parent = parent_id or self.parent_folder_id
        existing = self._list_children(parent, run_id)
        if existing:
            raise FileExistsError(f"Google Drive run folder already exists: {run_id}")
        metadata = {
            "name": run_id,
            "mimeType": self.folder_mime_type,
            "parents": [parent],
        }
        created = (
            self.service.files()
            .create(body=metadata, fields="id")
            .execute(num_retries=3)
        )
        return created["id"]

    def _ensure_folder(self, parent_id: str, name: str) -> str:
        existing = [
            item
            for item in self._list_children(parent_id, name)
            if item.get("mimeType") == self.folder_mime_type
        ]
        if len(existing) > 1:
            raise RuntimeError(f"Multiple Drive folders named {name!r} under {parent_id}")
        if existing:
            return existing[0]["id"]
        created = (
            self.service.files()
            .create(
                body={
                    "name": name,
                    "mimeType": self.folder_mime_type,
                    "parents": [parent_id],
                },
                fields="id",
            )
            .execute(num_retries=3)
        )
        return created["id"]

    def upload_file(self, path: str | Path, parent_id: str) -> dict[str, Any]:
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as error:
            raise RuntimeError("google-api-python-client is required") from error
        source = Path(path)
        media = MediaFileUpload(
            str(source),
            mimetype="application/octet-stream",
            chunksize=8 * 1024 * 1024,
            resumable=True,
        )
        request = self.service.files().create(
            body={"name": source.name, "parents": [parent_id]},
            media_body=media,
            fields="id,name,size",
        )
        response = None
        while response is None:
            _, response = request.next_chunk(num_retries=5)
        return {
            "drive_file_id": response["id"],
            "name": response["name"],
            "size_bytes": int(response.get("size", source.stat().st_size)),
            "sha256": sha256_file(source),
            "status": "uploaded",
        }

    def upload_run_directory(self, root: str | Path, run_id: str) -> dict[str, Any]:
        root_path = Path(root).resolve()
        run_folder_id = self.create_run_folder(run_id)
        folder_ids = {Path("."): run_folder_id}
        uploaded = []
        failures = []

        for path in sorted(root_path.rglob("*")):
            relative = path.relative_to(root_path)
            if path.is_dir():
                parent = folder_ids[relative.parent]
                folder_ids[relative] = self._ensure_folder(parent, path.name)
                continue
            parent = folder_ids[relative.parent]
            try:
                result = self.upload_file(path, parent)
                result["path"] = relative.as_posix()
                uploaded.append(result)
            except Exception as error:  # noqa: BLE001 - persist partial upload state
                self.logger.exception("Drive upload failed for %s", relative)
                failures.append({"path": relative.as_posix(), "error": type(error).__name__})
        return {
            "run_id": run_id,
            "drive_folder_id": run_folder_id,
            "status": "complete" if not failures else "partial",
            "uploaded": uploaded,
            "failures": failures,
        }

    def download_file(self, file_id: str, destination: str | Path) -> Path:
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as error:
            raise RuntimeError("google-api-python-client is required") from error
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        request = self.service.files().get_media(fileId=file_id)
        with destination_path.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=5)
        return destination_path

    def download_named_files(
        self,
        folder_id: str,
        names: set[str],
        destination_dir: str | Path,
    ) -> dict[str, Path]:
        children = self._list_children(folder_id)
        by_name = {item["name"]: item for item in children}
        missing = sorted(names - by_name.keys())
        if missing:
            raise FileNotFoundError(f"Drive folder is missing required files: {missing}")
        destination = Path(destination_dir)
        return {
            name: self.download_file(by_name[name]["id"], destination / name)
            for name in sorted(names)
        }
