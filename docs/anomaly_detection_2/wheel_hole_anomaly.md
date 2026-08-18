# Iniezione anomalie `hole` sulle ruote — v1

## Stato

La pipeline paired clean/anomaly e implementata e verificata sul pilot da 24
coppie. L'approvazione visiva delle quattro contact sheet T3 e completata.

- Sorgente immutabile:
  `outputs/anomaly_detection_2/pose_sampling/wheel_roll_pose_sampling.blend`
- SHA-256:
  `9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`
- Run canonico definitivo:
  `outputs/anomaly_detection_2/anomaly_batch/hole_pilot_v1_canonical_t3`
- Fingerprint run:
  `acd404b73a452fe3013112e03bf79b1e4b3681ee1d6ef78249733d1007c83db3`
- Probe A-B-A:
  `outputs/anomaly_detection_2/anomaly_batch/drift_probe_v1_t3_final/drift_report.json`
- Benchmark prestazioni:
  [anomaly_performance_benchmark.md](anomaly_performance_benchmark.md)

## Componenti autorevoli

- `configs/blender/wheel_hole_anomaly.json`: distribuzioni, dimensioni,
  geometria, materiali e gate.
- `configs/blender/anomaly_batch.json`: selezione stratificata del pilot,
  render e chunking.
- `src/wheel_preparation/hole_anomaly.py`: sampling e validazione pure Python.
- `src/wheel_preparation/anomaly_batch.py`: piano paired, pair-lock e
  fingerprint.
- `scripts/blender/wheel_hole_anomaly.py`: placement, cutout e carrier 3D.
- `scripts/blender/audit_anomaly_matrix.py`: audit geometrico 360 strati.
- `scripts/blender/render_anomaly_batch.py`: renderer persistente e gate.
- `scripts/host/run_anomaly_batch.py`: orchestrazione, resume e QA.
- `scripts/host/validate_anomaly_batch.py`: validator fail-closed.
- `scripts/host/run_anomaly_batch_drift_probe.py`: prova A-B-A.

CLI pubbliche:

```powershell
python scripts/host/run_anomaly_batch.py `
  --config configs/blender/anomaly_batch.json `
  --run-id <id> [--resume] [--blender-executable <path>]

python scripts/host/validate_anomaly_batch.py `
  --run-dir outputs/anomaly_detection_2/anomaly_batch/<id>
```

## Contratto del foro

Ogni membro anomalo contiene un solo danno etichettato `hole`. Il profilo e
un contorno irregolare deterministico appartenente a `jagged_slit`,
`branched_tear` o `peeled_window`. Le severita congelate del pilot sono small
55%, medium 35% e large 10%; il pilot usa 8 esempi per severita. La
distribuzione e 80% battistrada e 20% spalla esterna, con large limitato al
battistrada.

Per il dataset v1 il contratto approvato in
`dataset_composition_and_splits.md` aggiorna le severita a 50% small, 40%
medium e 10% large. La configurazione del pilot resta congelata per non
invalidare il run canonico.

La soluzione e ibrida:

1. il cutout shader apre la pelle originale senza Boolean distruttivi;
2. un carrier riusabile genera bordo e pareti aperte, senza cap incassato;
3. un flap sottile e ammesso soltanto nei medium/large selezionati;
4. il custom AOV `AnomalyMask` produce la maschera nello stesso render RGB.

La revisione passante rimuove ogni fondale sintetico: un ray-cast centrale
deve restare libero nel carrier per almeno 20 cm. L'RGB mostra quindi la
geometria realmente presente dietro la pelle della ruota. Il profilo attivo
`T3` conserva lo spessore fisico di 0,75 mm nella maggior parte del contorno e
genera due archi locali ripiegati, con transizione coseno deterministica dal
seed del profilo. La profondita massima e 2/3/4 mm per small/medium/large. Non
viene applicata uniformemente e non usa il precedente recess di cavita da
8--25 mm, che senza cap avrebbe formato un tunnel sintetico scuro. Il recess
campionato resta nei metadati come parametro sorgente non attivo. La maschera
include anche l'area vuota proiettando deterministicamente il contorno e
unendolo all'AOV di bordo, pareti e flap; non viene eseguito alcun render
aggiuntivo.

Il bordo dispone dei profili deterministici usati durante la revisione.
`R4_robust` e il profilo attivo:

| Profilo | Larghezza | Offset dalla pelle | Metallo esposto |
|---|---:|---:|---:|
| R0 current | 100% | 0,20 mm | 100% |
| R1 mild | 80% | 0,15 mm | 78% |
| R2 balanced | 60% | 0,10 mm | 58% |
| R3 strong | 45% | 0,06 mm | 40% |
| R4 minimal | 30% | 0,03 mm | 25% |
| R4 robust | 0,45–0,85 mm | 0,06 mm | 25% |

I valori percentuali moltiplicano il bordo e l'esposizione campionati per il
singolo foro; non cambiano apertura, placement o pair-lock fotografico.
R4 robust rimappa linearmente il campione sorgente nell'intervallo fisico
0,45–0,85 mm e mantiene l'esposizione minima di R4, ma usa l'offset di 0,06 mm
di R3 per ridurre il rischio di z-fighting quasi coplanare.

Il Boolean diretto e escluso perche ogni ruota contiene centinaia di
componenti disconnesse e migliaia di bordi aperti. I materiali originali
restano intatti: il runtime crea copie per ruota e non salva mai il `.blend`.

La spalla operativa e definita come la fascia cilindrica continua piu esterna,
non come la faccia laterale altamente perforata. Sul battistrada il ray cast
outside-in accetta soltanto la pelle sotto i grousers; sulla spalla accetta
solo il lato outboard. Nessun face index e codificato nel piano.

## Visibilita e pair-lock

Il centro e tutti i punti del contorno partecipano a facing, frame, settore
superiore e occlusion gate. Il placement prova al massimo 32 candidati senza
cambiare superficie, severita o settore. La camera prova al massimo 64 jitter
e puo modificare soltanto il jitter camera.

Il roll finale e anomaly-aware ed e condiviso dai due membri della coppia.
Sono condivisi anche camera, focale, luce, exposure, wear, matrici di rover e
ruota e allineamento alla patch. Le sole differenze ammesse sono cutout,
carrier, anomaly mask, artefatti e tempi.

## Maschere

- `target_wheel_mask`: superficie visibile della ruota target derivata dal
  membro clean e condivisa dalla coppia.
- `anomaly_mask`: apertura passante, bordo, pareti visibili e flap; le ombre
  sono escluse.
- Formato: PNG `L`, 1200x900, valori 0/255.

La sogliatura dell'AOV puo creare un'isola di quantizzazione sulla punta di un
contorno. Il renderer elimina soltanto componenti secondarie di massimo 4
pixel e registra `raw_component_count` e
`removed_raster_speckle_px`. Una componente separata piu grande non viene
corretta: il campione fallisce il gate. Nel pilot questa regola e intervenuta
una volta, su `dr_000088`, rimuovendo 2 pixel da una maschera principale di
1543 pixel.

## Verifica eseguita

Il run canonico definitivo ha prodotto:

- audit geometrico: 360/360 strati validi;
- 24 righe pair nel piano e nel manifest;
- 48 RGB, 24 target-wheel mask e 24 anomaly mask;
- 48 render call esatte;
- 2 aperture sorgente totali: preflight e unico chunk;
- validator finale `ok: true`;
- SHA-256 sorgente invariato.

## Approvazione visiva finale

Le contact sheet `A_overhead`, `C_leading_three_quarter`,
`C_trailing_three_quarter` e `D_upper_detail` del run canonico T3 sono state
riesaminate integralmente il 2026-08-13. In tutte le 24 coppie il foro e
leggibile nella posa assegnata, la posizione varia senza uscire dalla regione
visibile e il bordo R4 robusto non domina l'apertura. I casi passanti mostrano
geometria o terreno realmente presenti dietro la pelle della ruota; non e
presente un fondale sintetico. Il rosso compare esclusivamente negli overlay
QA, mai negli RGB. Contatto col terreno, crop intenzionale D, illuminazione e
usura risultano coerenti. Il pilot T3 e quindi il riferimento visivo definitivo
per il futuro bulk.

Tutte le 24 coppie hanno inoltre superato il gate di contatto col terreno e il
gate di apertura passante. La profondita minima registrata e 0,75 mm in tutte
le severita; le massime sono esattamente 2/3/4 mm per small/medium/large. La
minima mediana fotometrica e 10/255, la minima frazione di pixel cambiati nella
maschera e circa 0,664 e la minima energia locale e circa 0,99984, tutte entro
il contratto senza ridurre le soglie. Il run completo, incluso il preflight,
ha impiegato circa 1134 s sulla RX 6600.

Il resume definitivo ha ricontrollato 96 artefatti senza modificare alcun hash
o timestamp. Il probe `drift_a1 -> drift_b -> drift_a2` ha usato una sola
apertura e 6 render; RGB clean/anomaly e entrambe le maschere di A1/A2 sono
byte-identici, e coincidono anche matrici, sampling, geometria e gate.
I 24 RGB clean e le 24 target-wheel mask sono inoltre byte-identici al pilot
R4 robusto precedente: il carrier T3 disabilitato non introduce leakage nel
membro sano.

Due failure mode sono state trovate e risolte senza allentare i gate:

- su `dr_000084` la cavita in ombra aveva differenza mediana 8/255; una
  riflettanza PBR piu bassa, comunque non nera e senza emissione, porta il
  valore a 11/255 contro la soglia 10;
- su `dr_000088` una quantizzazione AOV generava l'isola di 2 pixel descritta
  sopra; il gate finale resta una sola componente.

La revisione del foro passante era stata verificata separatamente su sei pose
A, una per ruota; il pilot canonico definitivo ne estende ora la verifica a
tutte le 24 celle ruota/posa. Le immagini diagnostiche intermedie sono state
rimosse dopo la promozione di T3.

I profili del bordo sono stati confrontati su un foro small, uno medium su
shoulder e uno large. Tutte le 15 coppie hanno superato i gate; clean, target
mask, sampling e matrici sono identici tra i livelli. R4 robust e stato poi
verificato sugli stessi tre casi: larghezze effettive 0,516/0,714/0,730 mm,
3/3 gate completi superati e differenze rispetto a R4 minima confinate al
bordo. E il profilo attivo; le immagini diagnostiche intermedie sono state
rimosse.

Una review successiva della profondita visiva e disponibile in
`outputs/anomaly_detection_2/anomaly_batch/wall_depth_review`. Confronta, sugli
stessi stati fotografici di tre casi D small/medium/large:

- T1: parete fisica costante 0,75 mm;
- T2: due archi ripiegati irregolari fino a 1,5/2,25/3 mm;
- T3: due archi ripiegati irregolari fino a 2/3/4 mm.

I 18 render hanno superato i gate, i membri clean sono byte-identici fra i
livelli e la sorgente e rimasta invariata. T3 e stato scelto come variante
finale e promosso nel contratto di produzione. Le tavole `comparison_crops.png`
e `comparison_full.png` documentano la scelta; il successivo pilot canonico da
24 coppie e il drift probe confermano la stessa configurazione.

Durante il primo tentativo definitivo, `dr_000048` ha fallito in modo
deterministico con mediana 9/255. L'analisi ha mostrato che il carrier passante
conservava ancora le vecchie pareti di cavita profonde. La correzione alla
base fisica da 0,75 mm ha eliminato il tunnel artificiale e porta lo stesso
campione a 10/255 senza cambiare soglia, luce, posa, placement, severita o
larghezza R4. T3 estende soltanto due archi locali fino a 2/3/4 mm, senza
reintrodurre una parete profonda uniforme. Il tentativo fallito e documentato
qui come evidenza del comportamento fail-closed; i relativi artefatti parziali
sono stati rimossi dopo la promozione del pilot T3 canonico.

## Limiti

- Una sola classe: `hole`.
- Nessun Boolean, deformazione globale, grouser rotto o crepa isolata.
- Nessun depth/normal pass e nessuna anomaly mask per i clean standalone.
- Nessuna garanzia di pixel identity tra GPU/driver differenti.
- Il rapporto clean/anomaly e gli split sono definiti nel contratto dataset;
  render bulk ed export Anomalib/MVTec restano fuori scope di questa milestone.
- Le coppie dovranno restare nello stesso split futuro.
- Il render bulk non e ancora stato avviato.
