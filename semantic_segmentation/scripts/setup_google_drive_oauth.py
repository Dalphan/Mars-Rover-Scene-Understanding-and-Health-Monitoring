from __future__ import annotations

import argparse
import json
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authorize the S5Mars Kaggle notebooks, create their app-managed "
            "Google Drive folder, and write values to copy into Kaggle Secrets."
        )
    )
    parser.add_argument(
        "--client-secrets",
        type=Path,
        required=True,
        help="OAuth desktop client JSON downloaded from Google Cloud Console.",
    )
    parser.add_argument(
        "--folder-name",
        default="S5Mars_Experiments",
        help="Name of the app-managed folder created in My Drive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gdrive_kaggle_secrets.local.json"),
        help="Local secret bundle to copy into Kaggle Secrets and then delete.",
    )
    return parser.parse_args()


def create_folder(service, folder_name: str) -> str:
    escaped = folder_name.replace("\\", "\\\\").replace("'", "\\'")
    response = (
        service.files()
        .list(
            q=(
                f"name = '{escaped}' and "
                "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            ),
            spaces="drive",
            fields="files(id,name)",
        )
        .execute()
    )
    matches = response.get("files", [])
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple app-visible folders are named {folder_name!r}; "
            "rename extras and retry."
        )
    if matches:
        return matches[0]["id"]
    created = (
        service.files()
        .create(
            body={
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            },
            fields="id",
        )
        .execute()
    )
    return created["id"]


def main() -> None:
    args = parse_args()
    if not args.client_secrets.is_file():
        raise FileNotFoundError(args.client_secrets)
    flow = InstalledAppFlow.from_client_secrets_file(
        str(args.client_secrets),
        scopes=[DRIVE_FILE_SCOPE],
    )
    credentials = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
    )
    if not credentials.refresh_token:
        raise RuntimeError(
            "Google did not return a refresh token. Revoke the previous app grant "
            "and run setup again with prompt=consent."
        )
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    folder_id = create_folder(service, args.folder_name)

    client_config = json.loads(args.client_secrets.read_text(encoding="utf-8"))
    desktop = client_config.get("installed") or client_config.get("web")
    if not desktop:
        raise ValueError("Unsupported OAuth client JSON: missing installed/web section")
    secret_bundle = {
        "GDRIVE_CLIENT_ID": desktop["client_id"],
        "GDRIVE_CLIENT_SECRET": desktop["client_secret"],
        "GDRIVE_REFRESH_TOKEN": credentials.refresh_token,
        "GDRIVE_FOLDER_ID": folder_id,
    }
    args.output.write_text(
        json.dumps(secret_bundle, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created app-managed folder: {args.folder_name} ({folder_id})")
    print(f"Wrote Kaggle Secret values to: {args.output.resolve()}")
    print("Copy each value into Kaggle Secrets, then securely delete this local file.")
    print("Share the Drive folder manually with the second Google account.")


if __name__ == "__main__":
    main()
