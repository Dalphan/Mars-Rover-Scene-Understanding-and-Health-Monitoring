from omegaconf import OmegaConf

from mars_datasets.datamodule import SegmentationDataModule
from utils.runtime import configure_runtime


class CountingDataModule(SegmentationDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.build_calls = []

    def _build_dataset(self, split: str, is_train: bool):
        self.build_calls.append((split, is_train))
        return [split]


def test_datamodule_setup_skips_already_loaded_fit_stage():
    cfg = OmegaConf.create({"data": {"loader": {"batch_size": 1, "num_workers": 0}}})
    data_module = CountingDataModule(cfg)

    data_module.setup("fit")
    data_module.setup("fit")

    assert data_module.loaded_stages == {"fit"}
    assert data_module.build_calls == [("train", True), ("val", False)]


def test_configure_runtime_clamps_workers_and_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("utils.runtime.os.cpu_count", lambda: 2)
    monkeypatch.setattr("utils.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("utils.runtime.torch.cuda.device_count", lambda: 0)

    cfg = OmegaConf.create(
        {
            "data": {"loader": {"num_workers": "auto"}},
            "trainer": {"accelerator": "auto", "devices": "auto", "precision": "16-mixed"},
        }
    )

    runtime_info = configure_runtime(cfg)

    assert cfg.data.loader.num_workers == 1
    assert cfg.trainer.accelerator == "cpu"
    assert cfg.trainer.devices == 1
    assert cfg.trainer.precision == "32-true"
    assert runtime_info["workers"]["available_cpus"] == 2
