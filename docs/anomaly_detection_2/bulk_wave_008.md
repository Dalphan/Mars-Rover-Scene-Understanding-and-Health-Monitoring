# Bulk wave 008 — ulteriori 220 coppie test

## Stato operativo

Completata e validata il 17 agosto 2026 dopo la wave 007.

- run ID: `wave_008_pairs_220_test`;
- config: `configs/blender/bulk_waves/wave_008_pairs_220_test.json`;
- selezione: 220 coppie pending dello split test, offset 645;
- output atteso: 440 RGB, 220 target mask e 220 anomaly mask;
- PID iniziale: `10236`;
- log: `logs/anomaly_detection_2/bulk/wave_008_pairs_220_test.*.log`;
- nessuna sovrapposizione con wave 006 o 007;
- durata interna reale: circa 77 minuti;
- validator: 220/220 coppie, 440 RGB, 220 target mask e 220 anomaly mask;
- contrasto: 214 `normal_contrast`, 6 `low_contrast`.

Dopo il completamento restavano 110 coppie test, cioe 220 immagini, per
raggiungere le 10.000 immagini pianificate.
