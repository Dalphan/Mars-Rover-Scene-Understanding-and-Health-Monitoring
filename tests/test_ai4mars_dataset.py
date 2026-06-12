import numpy as np
from PIL import Image

from datasets.ai4mars import AI4MARSFolderDataset, audit_ai4mars


def test_ai4mars_dataset_discovers_images_masks_and_applies_range_mask(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    range_dir = tmp_path / "rng"
    images_dir.mkdir()
    masks_dir.mkdir()
    range_dir.mkdir()

    Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(images_dir / "sample.png")
    Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(masks_dir / "sample.png")
    Image.fromarray(np.array([[0, 1], [0, 0]], dtype=np.uint8)).save(range_dir / "sample.png")

    dataset = AI4MARSFolderDataset(root=tmp_path, taxonomy="core")
    sample = dataset[0]

    assert set(sample) == {"image", "mask", "dataset", "valid_classes", "image_id"}
    assert sample["image"].shape == (3, 2, 2)
    assert sample["mask"].dtype.is_floating_point is False
    assert sample["mask"].tolist() == [[1, 255], [1, 2]]
    assert sample["dataset"] == "ai4mars"
    assert sample["valid_classes"] == [0, 1, 2]
    assert sample["image_id"] == "sample"


def test_ai4mars_audit_reports_missing_masks(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(images_dir / "sample.png")
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(masks_dir / "sample.png")
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(images_dir / "missing.png")

    report = audit_ai4mars(tmp_path)

    assert report["images"] == 2
    assert report["masks"] == 1
    assert report["missing_masks"] == 1
    assert report["original_label_values"] == [0]
    assert report["mapped_label_values"] == [1]
