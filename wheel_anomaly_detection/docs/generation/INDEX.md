# Indice operativo — Anomaly Detection 2

Questo e il punto di ingresso per la pipeline Blender e il dataset in
`outputs/anomaly_detection_2`. Le note dei run conclusi sono state accorpate
nei documenti tecnici e nel riepilogo del dataset finale.

## Instradamento per argomento

| Se devi lavorare su | Leggi prima | Contenuto principale |
|---|---|---|
| Asset Curiosity, nomi ruote e gerarchia | [README.md](README.md) | Struttura semantica, assi, modifiche al GLB e limiti dell'asset |
| DTM/ortho HiRISE e terreno Gale | [gale_terrain.md](gale_terrain.md) | Sorgenti geospaziali, crop, CRS, texture e scena Gale |
| Patch base 4x4 m | [microterrain_level1.md](microterrain_level1.md) | Geometria deterministica e contratto Level 1 |
| Granelli, frammenti e rocce | [microterrain_level2.md](microterrain_level2.md) | Distribuzioni e Geometry Nodes Level 2 |
| Dettaglio fine del suolo | [microterrain_level3.md](microterrain_level3.md) | Polvere, albedo, roughness e bump Level 3 |
| Inquadrature delle ruote | [wheel_camera_pose_pilot.md](wheel_camera_pose_pilot.md) | Pose A/C1/C2/D, distanze, FOV e gate di framing |
| Roll e visibilita delle anomalie | [wheel_pose_sampling.md](wheel_pose_sampling.md) | Baseline, trasformazioni e vincoli anomaly-aware |
| Illuminazione marziana | [mars_lighting_pilot.md](mars_lighting_pilot.md) | Preset dusty/clear, ombre e jitter |
| Graffi e usura non anomala | [wheel_surface_wear_pilot.md](wheel_surface_wear_pilot.md) | Maschere light/evident e superfici interessate |
| Distribuzioni e jitter | [domain_randomization.md](domain_randomization.md) | Pesi, seed, pair-lock, retry camera e patch |
| Generazione automatica clean | [clean_batch_generator.md](clean_batch_generator.md) | Piano, manifest, RGB/mask, resume e validator |
| Fori paired clean/anomaly | [wheel_hole_anomaly.md](wheel_hole_anomaly.md) | Geometria, AOV mask, gate produttivi, pilot T3 e drift |
| Prestazioni clean e paired | [performance_and_validation.md](performance_and_validation.md) | Cache, pipeline CPU/GPU, regressioni e limiti dei confronti |
| Codifica PNG | [performance_and_validation.md#codifica-png](performance_and_validation.md#codifica-png) | Benchmark lossless e decisione sulla compressione |
| Composizione e split | [dataset_composition_and_splits.md](dataset_composition_and_splits.md) | Pool, 10.000 immagini, split e quote minime |
| Piano deterministico v1 | [bulk_plan_v1.md](bulk_plan_v1.md) | Indici globali, riuso pre-bulk e quote pre-render |
| Pre-bulk e audit quantitativo | [final_dataset_assembly.md#pre-bulk-riusato-nel-dataset-finale](final_dataset_assembly.md#pre-bulk-riusato-nel-dataset-finale) | 200 immagini riusate, tempi, distribuzioni e controlli raster |
| Provenienza delle wave completate | [final_dataset_assembly.md#provenienza-dei-run-completati](final_dataset_assembly.md#provenienza-dei-run-completati) | Run, split, conteggi, tempi e contrasto |
| Dataset finale e uso dei file | [final_dataset_assembly.md](final_dataset_assembly.md) | Hardlink, manifest, split, export e pulizia locale |

## Percorsi rapidi

- Generatore clean: [clean_batch_generator.md](clean_batch_generator.md),
  [domain_randomization.md](domain_randomization.md), poi
  [wheel_pose_sampling.md](wheel_pose_sampling.md) se cambia la posa.
- Fori o paired: [wheel_hole_anomaly.md](wheel_hole_anomaly.md), poi
  [domain_randomization.md](domain_randomization.md) e
  [wheel_pose_sampling.md](wheel_pose_sampling.md).
- Composizione o split: [dataset_composition_and_splits.md](dataset_composition_and_splits.md),
  [bulk_plan_v1.md](bulk_plan_v1.md) e
  [final_dataset_assembly.md](final_dataset_assembly.md).
- Terreno: [gale_terrain.md](gale_terrain.md), quindi i documenti
  microterrain Level 1, 2 e 3 nell'ordine della tabella.
- Modifiche visive: documenti di camera, illuminazione e usura.
- Nomi o gerarchia del rover: [README.md](README.md) e validator semantici.

## Stato autorevole corrente

- Branch di generazione: `anomaly-detection-2`.
- Sorgente clean immutabile:
  `outputs/anomaly_detection_2/pose_sampling/wheel_roll_pose_sampling.blend`,
  SHA-256 `9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`.
- Riferimenti clean verificati: `clean_batch/smoke_v2_terrain_contact_final`
  e `clean_batch/drift_probe_release`, sotto `outputs/anomaly_detection_2/`.
- Riferimenti paired verificati: `anomaly_batch/hole_pilot_v1_canonical_t3`
  e `anomaly_batch/drift_probe_v1_t3_final`. Le quattro contact sheet T3
  A/C1/C2/D sono approvate; il drift A-B-A e verificato.
- La produzione usa cache geometrica paired e pipeline CPU/GPU con coda
  limitata a 4; sul clean la cache e neutra. Misure e limiti sono nel
  [riepilogo prestazioni](performance_and_validation.md).
- Il bulk e completo: 8.750 unita e 10.000 RGB, split 7.000/1.000/2.000,
  train clean-only e coppie indivisibili in validation/test. Il pre-bulk
  di 150 clean e 25 coppie e incluso; non restano render pending.
- La vista canonica e
  `outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000`:
  10.000 RGB, 10.000 target-wheel mask e 1.250 anomaly mask, tramite hardlink
  verso i run immutabili. Il samples.csv originale e conservato nella root.
- Nel bulk il contrasto e diagnostico: normal_contrast e low_contrast sono
  registrati; no_effect e tutti i gate strutturali restano bloccanti.
  Soglie, motivazione e migrazione sono nel
  [contratto produttivo](wheel_hole_anomaly.md#gate-fotometrici-produttivi).
- La pulizia del 14 ottobre 2026 ha rimosso output legacy, copie QA,
  anteprime, backup .blend1, log e ambiente .venv. Dataset, run bulk/pre-bulk,
  scene correnti, pilot canonici e report sperimentali sono conservati.
  L'export ZIP Kaggle e stato rimosso, conservandone indice e metadati.
  I controlli hanno confermato 21.250 raster presenti e indici invariati.
- La successiva pulizia degli script ha rimosso 30 file della pipeline legacy,
  3 strumenti di analisi benchmark e 4 launcher PowerShell in background.
  Restano 63 file di generazione, inclusi i generatori correnti, la catena di
  ricostruzione della scena, i validator, le anteprime e gli script per le
  figure della tesi. I batch e l'assemblaggio si avviano con i comandi Python.

## Regola di manutenzione

Aggiornare questo indice quando cambia una milestone. Mantenere qui
instradamento e stato attuale; dettagli, misure e motivazioni restano nei
documenti specifici, evitando nuove pagine di solo avanzamento dei run.

L'archivio locale `docs/archive/legacy_blender` e stato rimosso durante la
pulizia autorizzata. Le versioni storiche restano recuperabili da Git.
