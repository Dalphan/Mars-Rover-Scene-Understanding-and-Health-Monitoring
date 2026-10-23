from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    output_dir,
    epoch,
    model,
    optimizer,
    scheduler,
    best_miou,
    cfg,
    is_best: bool,
    metadata: Mapping[str, Any] | None = None,
):
    """Write notebook-compatible ``last.ckpt`` and optional ``best.ckpt``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(metadata or {})
    config.setdefault(
        "hydra_config", OmegaConf.to_container(cfg, resolve=True)
    )
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "best_miou": best_miou,
        "config": config,
    }

    last_path = output_dir / "last.ckpt"
    torch.save(checkpoint, last_path)
    if is_best:
        torch.save(checkpoint, output_dir / "best.ckpt")
    return last_path


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def checkpoint_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Read current metadata while tolerating checkpoints from the old CLI."""

    raw = checkpoint.get("config")
    if raw is None:
        raw = checkpoint.get("cfg", {})
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    return dict(raw or {})


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    config = checkpoint_config(checkpoint)
    if not any(config.get(key) for key in ("dataset_name", "repo_id")):
        raise ValueError(
            "Checkpoint has no dataset_name or repo_id metadata; source "
            "compatibility cannot be verified safely"
        )
    mismatches = {
        key: {"checkpoint": config.get(key), "runtime": expected_value}
        for key, expected_value in expected.items()
        if config.get(key) is not None and config.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Checkpoint is incompatible with this run: {mismatches}")


def _get_secret(name: str) -> str:
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


def _build_drive_service():
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive checkpoint download requires the dependencies in requirements.txt"
        ) from exc

    credentials = Credentials(
        token=None,
        refresh_token=_get_secret("GDRIVE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=_get_secret("GDRIVE_CLIENT_ID"),
        client_secret=_get_secret("GDRIVE_CLIENT_SECRET"),
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def drive_query_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def download_checkpoint_from_drive_folder(
    folder_id: str | None,
    checkpoint_name: str,
    destination: str | Path,
) -> Path:
    if not folder_id:
        raise ValueError("No Google Drive folder ID is configured for this run")
    if checkpoint_name not in {"best.ckpt", "last.ckpt"}:
        raise ValueError(f"Unsupported checkpoint name: {checkpoint_name!r}")

    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive checkpoint download requires the dependencies in requirements.txt"
        ) from exc

    service = _build_drive_service()
    escaped_name = drive_query_escape(checkpoint_name)
    response = service.files().list(
        q=(
            f"'{folder_id}' in parents and name = '{escaped_name}' and "
            "mimeType != 'application/vnd.google-apps.folder' and trashed = false"
        ),
        fields="files(id,name,size,md5Checksum)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    matches = response.get("files", [])
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {checkpoint_name!r} in Drive folder "
            f"{folder_id}, found {len(matches)}"
        )

    remote = matches[0]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = service.files().get_media(
        fileId=remote["id"], supportsAllDrives=True
    )
    with temporary_path.open("wb") as output_file:
        downloader = MediaIoBaseDownload(
            output_file, request, chunksize=8 * 1024 * 1024
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()

    expected_size = int(remote["size"]) if remote.get("size") else None
    actual_size = temporary_path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        temporary_path.unlink(missing_ok=True)
        raise IOError(
            f"Incomplete checkpoint download: expected {expected_size} bytes, "
            f"got {actual_size}"
        )
    temporary_path.replace(destination)
    return destination
