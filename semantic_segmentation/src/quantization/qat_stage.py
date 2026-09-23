from __future__ import annotations

from src.quantization.qat_model import evaluate_torch_model
from src.quantization.qat_quantizers import prepare_selective_qat
from src.quantization.qat_storage import save_results
from src.quantization.qat_training import train_qat


def run_qat_stage(experiment, cfg, device):
    """Prepare selective quantizers, run QAT and update training reports."""

    student, quantizer_audit, tensor_quantizer_type, modelopt = (
        prepare_selective_qat(
            experiment.student,
            experiment.calibration_loader,
            cfg,
            device,
            experiment.paths.best_qat,
        )
    )
    results = experiment.results
    results["experiment"]["modelopt_quantizer_audit"] = quantizer_audit
    initialized_metrics = None
    if not cfg.steps.restore_saved_qat:
        initialized_metrics = evaluate_torch_model(
            student, experiment.val_loader, device, cfg, cfg.dataset.val_split
        )
        results["benchmarks"]["pytorch_modelopt_initialized_fake_quant"] = {
            "runtime": "pytorch",
            "precision": "fake_quant_int8_conv_mlp_decoder_linear",
            "metrics": initialized_metrics,
        }
    save_results(results, cfg, experiment.paths)

    student, history, best_miou, _ = train_qat(
        experiment.teacher,
        student,
        experiment.train_loader,
        experiment.val_loader,
        experiment.criterion,
        modelopt,
        experiment.paths,
        cfg,
        device,
    )
    experiment.student = student
    qat_metrics = evaluate_torch_model(
        student, experiment.val_loader, device, cfg, cfg.dataset.val_split
    )
    results["benchmarks"]["pytorch_modelopt_qat_fake_quant"] = {
        "runtime": "pytorch",
        "precision": "fake_quant_int8_conv_mlp_decoder_linear_qat",
        "metrics": qat_metrics,
    }
    if cfg.steps.restore_saved_qat:
        results["training"]["runtime_comparison_restore"] = {
            "restored_without_retraining": True,
            "best_val_miou_recomputed": best_miou,
            "best_checkpoint": str(experiment.paths.best_qat),
        }
    else:
        results["training"] = {
            "epochs_completed": len(history),
            "best_epoch": max(history, key=lambda row: row["val_miou"])["epoch"],
            "best_val_miou": best_miou,
            "history": history,
            "best_checkpoint": str(experiment.paths.best_qat),
            "pre_post_qat_validation": {
                "split": str(cfg.dataset.val_split),
                "before_qat": initialized_metrics,
                "after_qat": qat_metrics,
                "delta_after_minus_before": {
                    "pixel_accuracy": (
                        qat_metrics["pixel_accuracy"]
                        - initialized_metrics["pixel_accuracy"]
                    ),
                    "miou": qat_metrics["miou"] - initialized_metrics["miou"],
                },
            },
        }
    save_results(results, cfg, experiment.paths)
    return tensor_quantizer_type, quantizer_audit
