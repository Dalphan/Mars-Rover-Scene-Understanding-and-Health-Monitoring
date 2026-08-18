# Bulk wave 003 — 2.500 clean train

## Stato operativo

Avviata e completata il 16 agosto 2026 dopo la validazione delle wave 001 e
002. Ha prodotto 2.500 ulteriori immagini clean dello split train.

Validator finale: 2.500 RGB, 2.500 target mask, matrice 24/24 e stato
`complete`. Durata pipeline: 17.563,2 secondi, cioe circa 4 h 52 min 43 s.

- wave/run ID: `wave_003_train_clean_2500`;
- config: `configs/blender/bulk_waves/wave_003_train_clean_2500.json`;
- indici: `21550–24434`, 2.500 indici materializzati e disgiunti dalle wave
  precedenti;
- run: `outputs/anomaly_detection_2/bulk/waves/clean/wave_003_train_clean_2500`;
- log: `logs/anomaly_detection_2/bulk/wave_003_train_clean_2500.*.log`;
- 25 chunk da 100, esecuzione resumable in background;
- PID corrente registrato in
  `logs/anomaly_detection_2/bulk/wave_003_train_clean_2500.pid`.

Il numero di righe di `manifest.jsonl` e il conteggio autorevole durante il
run. La stima iniziale e circa 4 h 50 min, derivata dai 3.491,9 secondi
misurati per le 500 immagini della wave 002.

Resume:

```powershell
python scripts/host/run_clean_batch.py `
  --config configs/blender/bulk_waves/wave_003_train_clean_2500.json `
  --run-id wave_003_train_clean_2500 --resume `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe'
```

Al completamento, le tre wave conterranno 4.000 clean train; includendo le
150 immagini pre-bulk assegnate al train, il totale train disponibile salira
a 4.150.
