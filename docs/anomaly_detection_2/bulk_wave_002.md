# Bulk wave 002 — 500 clean train

## Stato operativo

Avviata e completata il 14 agosto 2026 dopo il completamento e la validazione
della wave 001. Ha prodotto le successive 500 immagini clean dello split train.

Validator finale: 500 RGB, 500 target mask, matrice 24/24 e stato `complete`.
Durata pipeline: 3.491,9 secondi, cioe circa 58 min 12 s.

- wave/run ID: `wave_002_train_clean_500`;
- config: `configs/blender/bulk_waves/wave_002_train_clean_500.json`;
- indici: `21029–21549`, con 500 indici materializzati e disgiunti dalla wave
  001;
- run: `outputs/anomaly_detection_2/bulk/waves/clean/wave_002_train_clean_500`;
- log: `logs/anomaly_detection_2/bulk/wave_002_train_clean_500.*.log`;
- 5 chunk da 100, esecuzione resumable in background;
- PID corrente registrato in
  `logs/anomaly_detection_2/bulk/wave_002_train_clean_500.pid`.

Il comando di resume e identico a quello della wave 001, sostituendo config e
run ID con `wave_002_train_clean_500`. Il numero di righe di `manifest.jsonl`
e il conteggio autorevole durante l'esecuzione.

La stima iniziale e 55–60 minuti, derivata dalle 1 h 57 min osservate per le
1.000 immagini della wave 001. Al completamento il train disponibile, incluso
il pre-bulk, salira a 1.650 immagini clean.
