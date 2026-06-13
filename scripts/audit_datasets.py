import json

from hydra import main

from mars_datasets.ai4mars import audit_ai4mars
from mars_datasets.mars_bench import audit_mars_bench


@main(version_base="1.3", config_path="../configs", config_name="config")
def run(cfg):
    data_cfg = cfg.data
    taxonomy = getattr(data_cfg, "taxonomy", "core")

    if data_cfg.name == "ai4mars":
        reports = [
            audit_ai4mars(root=data_cfg.root, split=getattr(getattr(data_cfg, split), "name", split), taxonomy=taxonomy)
            for split in ("train", "val", "test")
        ]
    elif data_cfg.name == "mars_bench":
        reports = [
            audit_mars_bench(
                dataset_name=data_cfg.hf_dataset,
                split=data_cfg.splits[split],
                taxonomy=taxonomy,
                image_key=getattr(data_cfg, "image_key", "image"),
                mask_key=getattr(data_cfg, "mask_key", "mask"),
                config_name=getattr(data_cfg, "hf_config", None),
                max_samples=getattr(data_cfg, "audit_max_samples", None),
            )
            for split in ("train", "val", "test")
        ]
    else:
        raise ValueError(f"Unknown dataset name for audit: {data_cfg.name}")

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    run()
