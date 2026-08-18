# Bulk wave 001 — 1.000 clean train

## Stato operativo

Avviata e completata il 14 agosto 2026 come primo blocco del bulk v1. Il job ha prodotto 1.000
immagini clean dello split train, selezionate esclusivamente dalle unita
`pending` del piano completo.

Validator finale: 1.000 RGB, 1.000 target mask, matrice 24/24 e stato
`complete`. Durata osservata circa 1 h 57 min.

- wave ID e run ID: `wave_001_train_clean_1000`;
- config: `configs/blender/bulk_waves/wave_001_train_clean_1000.json`;
- run: `outputs/anomaly_detection_2/bulk/waves/clean/wave_001_train_clean_1000`;
- log stdout/stderr: `logs/anomaly_detection_2/bulk/`;
- manifest della selezione:
  `outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000/planning/bulk_v1/waves/wave_001_train_clean_1000.json`;
- 10 chunk da 100, pipeline CPU/GPU asincrona, PNG lossless zlib 4;
- processo avviato in background; il PID corrente e registrato nel file
  `logs/anomaly_detection_2/bulk/wave_001_train_clean_1000.pid`.

Il conteggio autorevole durante il run e il numero di righe complete in
`manifest.jsonl`; `run.json` aggiorna il totale soprattutto ai confini di
chunk e al completamento.

## Resume

In caso di chiusura o crash:

```powershell
python scripts/host/run_clean_batch.py `
  --config configs/blender/bulk_waves/wave_001_train_clean_1000.json `
  --run-id wave_001_train_clean_1000 --resume `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe'
```

Il resume salta soltanto le righe gia committate e con fingerprint coerente.
Artefatti non registrati o parziali non vengono considerati completi.

## Scelta dello split

Il train clean e stato scelto per la prima wave perche non richiede preflight
dei fori, e il blocco piu semplice da interrompere e riprendere e aumenta
subito il supporto del training. Le 150 clean pre-bulk gia assegnate al train
restano separate: al termine della wave il train disponibile salira a 1.150
immagini, senza duplicati di indice.
