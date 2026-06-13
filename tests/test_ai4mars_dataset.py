import numpy as np
from PIL import Image

from mars_datasets.ai4mars import AI4MARSFolderDataset, audit_ai4mars


def _write_ai4mars_msl_fixture(root):
    msl = root / "A)4MARS" / "ai4mars-dataset-merged-0.6" / "msl"

    mcam_images = msl / "mcam" / "images"
    mcam_labels = msl / "mcam" / "labels" / "train"
    ncam_images = msl / "ncam" / "images" / "edr"
    ncam_rover = msl / "ncam" / "images" / "mxy"
    ncam_range = msl / "ncam" / "images" / "rng-30m"
    ncam_train = msl / "ncam" / "labels" / "train"
    ncam_test = msl / "ncam" / "labels" / "test" / "masked-gold-min3-100agree"

    for path in (mcam_images, mcam_labels, ncam_images, ncam_rover, ncam_range, ncam_train, ncam_test):
        path.mkdir(parents=True)

    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    Image.fromarray(image).save(mcam_images / "mcam_sample.JPG")
    Image.fromarray(mask).save(mcam_labels / "mcam_sample_merged.png")

    Image.fromarray(image).save(ncam_images / "ncam_sample.JPG")
    Image.fromarray(mask).save(ncam_train / "ncam_sample_merged.png")
    Image.fromarray(np.array([[0, 1], [0, 0]], dtype=np.uint8)).save(ncam_range / "ncam_sample.png")
    Image.fromarray(np.array([[0, 0], [1, 0]], dtype=np.uint8)).save(ncam_rover / "ncam_sample.png")

    Image.fromarray(image).save(ncam_images / "ncam_test.JPG")
    Image.fromarray(mask).save(ncam_test / "ncam_test_merged.png")

    return msl


def test_ai4mars_dataset_loads_mcam_and_ncam_train_layout(tmp_path):
    msl = _write_ai4mars_msl_fixture(tmp_path)

    dataset = AI4MARSFolderDataset(root=msl, split="train", taxonomy="core")
    mcam_sample = dataset[0]
    ncam_sample = dataset[1]

    assert len(dataset) == 2
    assert mcam_sample["image"].shape == (3, 2, 2)
    assert mcam_sample["mask"].tolist() == [[1, 0], [1, 2]]
    assert ncam_sample["mask"].tolist() == [[1, 255], [255, 2]]
    assert ncam_sample["dataset"] == "ai4mars"
    assert ncam_sample["valid_classes"] == [0, 1, 2]
    assert ncam_sample["image_id"] == "ncam_sample"


def test_ai4mars_audit_uses_direct_msl_layout(tmp_path):
    _write_ai4mars_msl_fixture(tmp_path)

    train_report = audit_ai4mars(tmp_path, split="train")
    test_report = audit_ai4mars(tmp_path, split="test")

    assert train_report["images"] == 2
    assert train_report["masks"] == 2
    assert train_report["missing_masks"] == 0
    assert train_report["original_label_values"] == [0, 1, 2, 3]
    assert train_report["mapped_label_values"] == [0, 1, 2, 255]
    assert test_report["images"] == 1
