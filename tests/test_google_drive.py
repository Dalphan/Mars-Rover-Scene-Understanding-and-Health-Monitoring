from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.google_drive import GoogleDriveRunStore, _escape_query_value


class _Executable:
    def __init__(self, result):
        self.result = result

    def execute(self, num_retries=0):
        return self.result


class _Files:
    def __init__(self):
        self.items = []
        self.created = []

    def list(self, **kwargs):
        return _Executable({"files": list(self.items)})

    def create(self, body, fields, media_body=None):
        created = {
            "id": f"id-{len(self.created) + 1}",
            "name": body["name"],
            "size": "0",
        }
        self.created.append((body, media_body))
        return _Executable(created)


class _Service:
    def __init__(self):
        self.resource = _Files()

    def files(self):
        return self.resource


def test_drive_query_escaping_and_no_overwrite():
    assert _escape_query_value("a'b\\c") == "a\\'b\\\\c"
    service = _Service()
    store = GoogleDriveRunStore(service, "parent")
    assert store.create_run_folder("run-1") == "id-1"
    service.resource.items = [
        {"id": "existing", "name": "run-1", "mimeType": store.folder_mime_type}
    ]
    with pytest.raises(FileExistsError):
        store.create_run_folder("run-1")


def test_drive_store_requires_parent():
    with pytest.raises(ValueError):
        GoogleDriveRunStore(_Service(), "")
