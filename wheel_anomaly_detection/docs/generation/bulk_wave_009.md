# Bulk wave 009 — ultime 110 coppie test

## Stato operativo

Wave finale completata e validata il 18 agosto 2026 dopo la wave 008.

- run ID: `wave_009_pairs_110_test_final`;
- config: `configs/blender/bulk_waves/wave_009_pairs_110_test_final.json`;
- selezione: ultime 110 coppie pending dello split test, offset 865;
- output atteso: 220 RGB, 110 target mask e 110 anomaly mask;
- PID iniziale: `10204`;
- log: `logs/anomaly_detection_2/bulk/wave_009_pairs_110_test_final.*.log`;
- nessuna sovrapposizione fra le wave 006–009;
- durata interna reale: circa 41 minuti;
- validator: 110/110 coppie, 220 RGB, 110 target mask e 110 anomaly mask;
- contrasto: 109 `normal_contrast`, 1 `low_contrast`.

Non restano unita da renderizzare nel piano da 10.000 immagini. Restano da
eseguire assemblaggio del pool finale,
verifica globale di split/quote/integrita e materializzazione della struttura
dataset destinata al training.
