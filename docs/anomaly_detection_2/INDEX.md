# Indice operativo — Anomaly Detection 2

Questo e il primo file da consultare per qualsiasi lavoro sulla pipeline
Blender in `outputs/anomaly_detection_2`. Serve a trovare la fonte corretta
senza rileggere tutti i documenti o affidarsi a note ormai superate.

## Instradamento per argomento

| Se devi lavorare su | Leggi prima | Contenuto principale |
|---|---|---|
| Asset Curiosity, nomi ruote e gerarchia | [README.md](README.md) | Struttura semantica, assi, modifiche al GLB e limiti dell'asset |
| DTM/ortho HiRISE e terreno Gale | [gale_terrain.md](gale_terrain.md) | Sorgenti geospaziali, crop, CRS, texture e scena Gale |
| Patch base 4x4 m | [microterrain_level1.md](microterrain_level1.md) | Geometria deterministica e contratto Level 1 |
| Granelli, frammenti e rocce | [microterrain_level2.md](microterrain_level2.md) | Distribuzioni, classi dimensionali e Geometry Nodes Level 2 |
| Dettaglio fine del suolo | [microterrain_level3.md](microterrain_level3.md) | Polvere, albedo, roughness e bump Level 3 |
| Inquadrature delle ruote | [wheel_camera_pose_pilot.md](wheel_camera_pose_pilot.md) | Pose A/C1/C2/D, distanze, FOV e gate di framing |
| Rotazione ruote e visibilita futura anomalie | [wheel_pose_sampling.md](wheel_pose_sampling.md) | Roll, trasformazioni baseline e vincoli anomaly-aware |
| Illuminazione marziana | [mars_lighting_pilot.md](mars_lighting_pilot.md) | Preset dusty/clear, ombre L5 e jitter luce |
| Graffi e usura non anomala | [wheel_surface_wear_pilot.md](wheel_surface_wear_pilot.md) | Maschere light/evident, proiezione e superfici interessate |
| Distribuzioni e jitter complessivi | [domain_randomization.md](domain_randomization.md) | Pesi, seed, pair-lock, retry camera e allineamento alla patch |
| Generazione automatica clean | [clean_batch_generator.md](clean_batch_generator.md) | Config, piano, manifest, pass RGB/mask, resume e validator |
| Prestazioni renderer clean | [clean_performance_benchmark.md](clean_performance_benchmark.md) | Cache geometrica e pipeline sovrapposta CPU/GPU |
| Fori paired clean/anomaly | [wheel_hole_anomaly.md](wheel_hole_anomaly.md) | Sampling, geometria ibrida, AOV mask, gate, pilot e drift |
| Prestazioni renderer paired | [anomaly_performance_benchmark.md](anomaly_performance_benchmark.md) | Benchmark 24 coppie, cache geometrica, profiling e pipeline CPU/GPU |
| Composizione e split dataset | [dataset_composition_and_splits.md](dataset_composition_and_splits.md) | Pool canonico, 10.000 immagini, split 70/10/20 e quote minime |
| Pre-bulk da 200 immagini | [prebulk_200.md](prebulk_200.md) | Piano stratificato, run validati, tempi reali e pool parziale riusabile |
| Audit pre-bulk e benchmark PNG | [prebulk_quantitative_audit.md](prebulk_quantitative_audit.md) | Integrita raster, distribuzioni, duplicati, margini gate e costo reale della codifica PNG |
| Piano completo da 10.000 immagini | [bulk_plan_v1.md](bulk_plan_v1.md) | Indici globali, riuso pre-bulk, split pre-render, quote esatte e config dei 9.800 rimanenti |
| Prima wave bulk da 1.000 | [bulk_wave_001.md](bulk_wave_001.md) | Run clean train in background, log, output e comando di resume |
| Seconda wave bulk da 500 | [bulk_wave_002.md](bulk_wave_002.md) | Secondo blocco clean train, indici, log e resume |
| Terza wave bulk da 2.500 | [bulk_wave_003.md](bulk_wave_003.md) | Terzo blocco clean train, 25 chunk, log e resume |
| Quarta wave bulk da 2.200 | [bulk_wave_004.md](bulk_wave_004.md) | Quarto blocco clean train, 22 chunk, log e resume |
| Sequenza bulk da 2.500 | [bulk_sequence_005.md](bulk_sequence_005.md) | Residuo clean train/validation e 675 coppie validation/test in esecuzione seriale |
| Wave paired test da 220 coppie | [bulk_wave_007.md](bulk_wave_007.md) | Run da circa 90 minuti, 440 immagini test e monitoraggio |
| Seconda wave paired test da 220 | [bulk_wave_008.md](bulk_wave_008.md) | Ulteriori 440 immagini test; restano 110 coppie |
| Wave finale da 110 coppie | [bulk_wave_009.md](bulk_wave_009.md) | Ultimi 220 render pianificati e passaggio all'assemblaggio finale |
| Dataset finale assemblato | [final_dataset_assembly.md](final_dataset_assembly.md) | Vista ordinata hardlink, manifest, split, conteggi e regole d'uso |

## Percorsi rapidi

- Modifiche al generatore batch: leggere nell'ordine
  [clean_batch_generator.md](clean_batch_generator.md),
  [domain_randomization.md](domain_randomization.md) e, se cambia la geometria
  della posa, [wheel_pose_sampling.md](wheel_pose_sampling.md).
- Modifiche a fori, maschere anomalia o batch paired: leggere
  [wheel_hole_anomaly.md](wheel_hole_anomaly.md), poi
  [domain_randomization.md](domain_randomization.md) e
  [wheel_pose_sampling.md](wheel_pose_sampling.md).
- Modifiche alla composizione bulk o agli split: leggere
  [dataset_composition_and_splits.md](dataset_composition_and_splits.md), poi
  [clean_batch_generator.md](clean_batch_generator.md) e
  [wheel_hole_anomaly.md](wheel_hole_anomaly.md).
- Modifiche al terreno: leggere [gale_terrain.md](gale_terrain.md), poi i tre
  documenti microterrain nell'ordine Level 1, Level 2, Level 3.
- Modifiche visive alle ruote: leggere prima il documento della camera, poi
  illuminazione e usura.
- Modifiche ai nomi o alla gerarchia del rover: partire sempre da
  [README.md](README.md) e rieseguire i validator semantici.

## Stato autorevole corrente

- Branch: `anomaly-detection-2`.
- Sorgente clean immutabile:
  `outputs/anomaly_detection_2/pose_sampling/wheel_roll_pose_sampling.blend`.
- SHA-256 sorgente:
  `9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`.
- Run clean verificato:
  `outputs/anomaly_detection_2/clean_batch/smoke_v2_terrain_contact_final`.
- Probe di drift verificato:
  `outputs/anomaly_detection_2/clean_batch/drift_probe_release`.
- Pilot paired `hole` definitivo verificato (foro passante, contatto terreno,
  bordo R4 robusto e pareti a pieghe locali T3):
  `outputs/anomaly_detection_2/anomaly_batch/hole_pilot_v1_canonical_t3`.
- Approvazione visiva T3 completata sulle quattro contact sheet A/C1/C2/D.
- Probe A-B-A paired definitivo verificato:
  `outputs/anomaly_detection_2/anomaly_batch/drift_probe_v1_t3_final`.
- Benchmark paired completato: `cache_64` e la candidata produttiva perche
  conserva RGB e mask byte-identici; la promozione definitiva resta
  subordinata al pre-bulk.
- Regressione paired post-pipeline verificata su 24 coppie:
  `outputs/anomaly_detection_2/benchmark/anomaly_24/cache_64_regression_verified`.
- Benchmark clean completato: la cache e neutra; la pipeline CPU/GPU con coda
  limitata a 4 riduce del 33,75% il tempo interno ed e abilitata nel config
  clean, con output byte-identici. Resta da confermare sul pre-bulk.
- Composizione approvata: 10.000 immagini, split 70/10/20, train clean-only e
  coppie indivisibili in validation/test.
- Pre-bulk completato: 150 clean standalone e 25 coppie, per 200 immagini
  complessive, indicizzate nel pool parziale `prebulk_200_v1`. Tempo reale tra
  i timestamp dei due run: 29 min 43 s. Il bulk restante non e stato avviato.
- Audit quantitativo pre-bulk superato su 400 raster senza errori, artefatti
  orfani o duplicati. Benchmark PNG concluso: zlib level 1 non viene promosso,
  perche il risparmio end-to-end stimato e inferiore allo 0,4% a fronte di
  circa il 59% di spazio in piu.
- Piano bulk v1 materializzato: 8.750 unita/10.000 immagini, delle quali 200
  gia prodotte e 9.800 pending. Split, indici e quote congiunte sono fissati
  prima dei render; tutte le celle di evaluation raggiungono il minimo 1 in
  validation e 2 in test.
- Bulk wave 001 completata e validata: 1.000 RGB e 1.000 target mask clean
  train, matrice 24/24, durata circa 1 h 57 min.
- Bulk wave 002 completata e validata: ulteriori 500 clean train in circa
  58 min 12 s.
- Bulk wave 003 completata e validata: 2.500 clean train in circa 4 h 52 min
  43 s.
- Bulk wave 004 completata e validata: 2.200 clean train in circa 4 h 16 min
  55 s. Le prime quattro wave contengono complessivamente 6.200 clean train.
- Bulk sequence 005 avviata: 1.150 clean residue e 675 coppie, per 2.500
  immagini. I due step sono seriali e resumable; al termine train e validation
  saranno completi e resteranno 1.100 immagini paired test.
- Il blocco clean della sequence 005 e completo e validato. Due tentativi
  paired hanno esposto un gate mediano ridondante e fragile (`9/255` e poi
  `8/255`, con oltre l'86% della mask gia sopra `6/255`). Le directory fallite
  sono state rimosse e il run definitivo `wave_006_pairs_675_final` riparte
  con mediana coerente a `6/255`; tutti gli altri gate restano invariati.
- Il successivo tentativo paired si e fermato a `changed_fraction=0,6394`
  contro `0,65`. La soglia produttiva e stata fissata a `0,58` su decisione
  esplicita; il run sostitutivo da zero e `wave_006_pairs_675_cf58`.
- Il run `cf58` ha mostrato un caso strutturalmente valido a basso contrasto.
  I gate fotometrici sono ora diagnostici (`normal_contrast`/`low_contrast`),
  mentre `no_effect` e tutti i gate strutturali restano bloccanti. Il run
  sostitutivo e `wave_006_pairs_675_photometric_diagnostic`.
- Il run diagnostico e stato ripreso in sicurezza dopo un crash del PC: nessuna
  coppia risultava committata e gli artefatti orfani vengono rigenerati; il
  fingerprint del resume coincide con piano, configurazioni e codice correnti.
- Run diagnostico completato e validato: 675/675 coppie, 665 normal-contrast e
  10 low-contrast.
- Wave 007 completata e validata: 220/220 coppie, 214 normal-contrast e 6
  low-contrast, durata interna circa 77 minuti.
- Wave 008 completata e validata: 220/220 coppie, 214 normal-contrast e 6
  low-contrast, durata interna circa 77 minuti.
- Wave 009 finale completata e validata: 110/110 coppie, 109 normal-contrast e
  1 low-contrast, durata interna circa 41 minuti. Non restano render pending.
- Dataset finale assemblato e verificato in
  `outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000`: 8.750
  unita, 10.000 RGB, 10.000 target-wheel mask e 1.250 anomaly mask. La vista
  ordinata usa hardlink verso i run immutabili e rispetta gli split
  7.000/1.000/2.000 senza duplicare lo spazio dei raster.
- Pulizia post-assemblaggio completata: rimossi benchmark raster, revisioni
  visive superate, smoke v1, dipendenze generate e log/PID falliti. Run bulk,
  pre-bulk, pilot T3, probe di drift e sorgenti riproducibili sono conservati.

## Regola di manutenzione

Quando viene aggiunta una nuova milestone o un documento diventa obsoleto,
aggiornare prima questo indice. I dettagli tecnici devono restare nel documento
specifico: qui vanno mantenuti soltanto instradamento e stato autorevole.

I file in [`docs/handoff`](../handoff/README.md) sono un archivio della
pipeline legacy: non devono essere letti o aggiornati per le normali milestone
di `anomaly_detection_2`, salvo che serva ricostruire una decisione storica.
