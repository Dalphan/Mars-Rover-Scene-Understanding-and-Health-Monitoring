# Bulk wave 007 — 220 coppie test

## Stato operativo

Completata e validata il 17 agosto 2026 dopo il completamento del run
`wave_006_pairs_675_photometric_diagnostic`.

- run ID: `wave_007_pairs_220_test`;
- config: `configs/blender/bulk_waves/wave_007_pairs_220_test.json`;
- selezione: 220 coppie pending dello split test, offset 425;
- output atteso: 440 RGB, 220 target mask e 220 anomaly mask;
- PID iniziale: `5940`;
- log: `logs/anomaly_detection_2/bulk/wave_007_pairs_220_test.*.log`;
- durata reale interna: circa 77 minuti;
- validator: 220/220 coppie, 440 RGB, 220 target mask e 220 anomaly mask;
- contrasto: 214 `normal_contrast`, 6 `low_contrast`.

La selezione non si sovrappone alle 425 coppie test della wave 006. Dopo questa
wave restavano 330 coppie test, cioe 660 immagini, per completare il dataset
pianificato.

Il validator usa ora accesso raster bulk via byte buffer, semanticamente
equivalente al precedente ciclo `getpixel`, per evitare decine di minuti CPU
spesi soltanto a scansionare le maschere.
