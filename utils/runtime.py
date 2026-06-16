from __future__ import annotations

import os
from typing import Any

import torch
from omegaconf import open_dict


def configure_runtime(cfg) -> dict[str, Any]:
    """Clamp worker and device settings to what the current machine can run."""
    worker_info = _resolve_num_workers(_cfg_get(cfg.data.loader, "num_workers", "auto"))
    accelerator, devices, gpu_info = _resolve_trainer_devices(
        accelerator=_cfg_get(cfg.trainer, "accelerator", "auto"),
        devices=_cfg_get(cfg.trainer, "devices", "auto"),
    )
    precision = _resolve_precision(
        accelerator=accelerator,
        precision=_cfg_get(cfg.trainer, "precision", "32-true"),
    )

    with open_dict(cfg):
        cfg.data.loader.num_workers = worker_info["selected"]
        cfg.trainer.accelerator = accelerator
        cfg.trainer.devices = devices
        cfg.trainer.precision = precision

    print(
        "[runtime] dataloader workers: "
        f"requested={worker_info['requested']} available_cpus={worker_info['available_cpus']} "
        f"selected={worker_info['selected']}"
    )
    print(
        "[runtime] trainer devices: "
        f"requested={gpu_info['requested_devices']} available_gpus={gpu_info['available_gpus']} "
        f"accelerator={accelerator} devices={devices} precision={precision}"
    )
    if gpu_info["device_names"]:
        print(f"[runtime] GPU names: {', '.join(gpu_info['device_names'])}")

    return {"workers": worker_info, "devices": gpu_info}


def _resolve_num_workers(requested) -> dict[str, Any]:
    available_cpus = os.cpu_count() or 1
    max_workers = max(0, available_cpus - 1)

    if _is_auto(requested):
        selected = min(4, max_workers)
    else:
        selected = min(max(0, int(requested)), max_workers)

    return {
        "requested": requested,
        "available_cpus": available_cpus,
        "selected": selected,
    }


def _resolve_trainer_devices(accelerator, devices) -> tuple[str, int | list[int], dict[str, Any]]:
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested_accelerator = str(accelerator).lower() if accelerator is not None else "auto"
    requested_devices = devices
    device_names = [torch.cuda.get_device_name(index) for index in range(available_gpus)]

    wants_gpu = requested_accelerator in {"auto", "gpu", "cuda"}
    if wants_gpu and available_gpus > 0:
        selected_devices = _select_gpu_devices(devices, available_gpus)
        return (
            "gpu",
            selected_devices,
            {
                "requested_accelerator": accelerator,
                "requested_devices": requested_devices,
                "available_gpus": available_gpus,
                "device_names": device_names,
            },
        )

    return (
        "cpu",
        1,
        {
            "requested_accelerator": accelerator,
            "requested_devices": requested_devices,
            "available_gpus": available_gpus,
            "device_names": device_names,
        },
    )


def _select_gpu_devices(devices, available_gpus: int) -> int | list[int]:
    if _is_auto(devices) or devices == -1:
        return available_gpus
    if isinstance(devices, (list, tuple)):
        selected = [int(device) for device in devices if 0 <= int(device) < available_gpus]
        return selected or 1
    return max(1, min(int(devices), available_gpus))


def _resolve_precision(accelerator: str, precision):
    if accelerator == "cpu" and str(precision).startswith("16"):
        return "32-true"
    return precision


def _is_auto(value) -> bool:
    return value is None or (isinstance(value, str) and value.lower() == "auto")


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    return getattr(cfg, key, default)
