# Bulk wave 004 — 2.200 clean train

## Stato operativo

Completata e validata il 16 agosto 2026 dopo la wave 003. Ha prodotto 2.200
ulteriori immagini clean dello split train.

- wave/run ID: `wave_004_train_clean_2200`;
- config: `configs/blender/bulk_waves/wave_004_train_clean_2200.json`;
- indici: `24437–29383`, 2.200 indici materializzati e disgiunti dalle wave
  precedenti;
- run: `outputs/anomaly_detection_2/bulk/waves/clean/wave_004_train_clean_2200`;
- log: `logs/anomaly_detection_2/bulk/wave_004_train_clean_2200.*.log`;
- 22 chunk da 100;
- 2.200 righe manifest, 2.200 RGB e 2.200 target mask;
- validator superato, matrice camera/ruota 24/24;
- sorgente invariata prima e dopo il run;
- durata reale: 15.415,4 s, cioe circa 4 h 16 min 55 s.

Il numero di righe di `manifest.jsonl` resta il conteggio autorevole del run.

Resume:

```powershell
python scripts/host/run_clean_batch.py `
  --config configs/blender/bulk_waves/wave_004_train_clean_2200.json `
  --run-id wave_004_train_clean_2200 --resume `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe'
```

Le quattro wave contengono 6.200 clean train. Includendo le 150 immagini
pre-bulk assegnate al train, il totale disponibile e 6.350; restano 650 clean
train per chiudere il train da 7.000.
