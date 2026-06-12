import numpy as np
from PIL import Image

from datasets import mars_bench
from datasets.mars_bench import MarsBenchHFDataset


class FakeHFDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def test_mars_bench_dataset_maps_seg_labels_without_network(monkeypatch):
    records = [
        {
            "image": Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)),
            "mask": Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)),
            "image_id": "abc",
        }
    ]
    monkeypatch.setattr(mars_bench, "_load_hf_dataset", lambda *args, **kwargs: FakeHFDataset(records))

    dataset = MarsBenchHFDataset("Mirali33/mb-mars_seg_msl", split="train")
    sample = dataset[0]

    assert sample["image"].shape == (3, 2, 2)
    assert sample["mask"].tolist() == [[255, 0], [1, 2]]
    assert sample["dataset"] == "mb-mars_seg_msl"
    assert sample["valid_classes"] == [0, 1, 2]
    assert sample["image_id"] == "abc"


def test_mars_bench_dataset_uses_s5mars_mapping(monkeypatch):
    records = [
        {
            "image": np.zeros((1, 3, 3), dtype=np.uint8),
            "mask": np.array([[1, 4, 6]], dtype=np.uint8),
        }
    ]
    monkeypatch.setattr(mars_bench, "_load_hf_dataset", lambda *args, **kwargs: FakeHFDataset(records))

    dataset = MarsBenchHFDataset("Mirali33/mb-s5mars", split="train")

    assert dataset[0]["mask"].tolist() == [[0, 2, 1]]
