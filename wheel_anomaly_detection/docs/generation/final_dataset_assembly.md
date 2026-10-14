# Assemblaggio finale del dataset v1

Stato: completato e verificato il 18 agosto 2026.

## Risultato

Il dataset canonico e disponibile in:

```text
outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000/
```

La directory contiene una vista ordinata dei raster prodotti dai run, senza
duplicarne i byte. Gli artefatti sono materializzati tramite hardlink: il file
nella cartella dataset e il file nel run sorgente sono due nomi dello stesso
contenuto sul disco.

```text
curiosity_wheel_hole_v1_10000/
├── dataset.json
├── image_manifest.jsonl
├── samples.csv
├── images/
│   ├── train/clean/
│   ├── validation/{clean,hole}/
│   └── test/{clean,hole}/
├── masks/
│   ├── target_wheel/<split>/<condition>/
│   └── anomaly/{validation,test}/hole/
├── pool/manifest.jsonl
├── splits/{train,validation,test}.jsonl
└── planning/bulk_v1/
```

## Conteggi verificati

| Split | Clean | Hole | Totale immagini |
|---|---:|---:|---:|
| train | 7.000 | 0 | 7.000 |
| validation | 750 | 250 | 1.000 |
| test | 1.000 | 1.000 | 2.000 |
| totale | 8.750 | 1.250 | 10.000 |

Sono presenti inoltre:

- 10.000 RGB;
- 10.000 target-wheel mask;
- 1.250 anomaly mask;
- 8.750 righe nel manifest delle unita;
- 10.000 righe nel manifest per immagine;
- 7.000/1.000/2.000 righe negli indici train/validation/test.

L'assemblatore ha verificato 20.000 artefatti sorgente tramite SHA-256 prima
della pubblicazione. Il riepilogo e gli hash dei manifest e degli split sono
registrati in `dataset.json`.

## Pre-bulk riusato nel dataset finale

Il pre-bulk e stato completato e validato il 14 agosto 2026. Le sue 175
unita comprendono 150 clean standalone e 25 coppie clean/hole, per 200 RGB.
I 400 raster canonici unici comprendono 200 RGB, 175 target-wheel mask e
25 anomaly mask, per 212,95 MiB. La target mask di una coppia e condivisa
dai due membri. Nel dataset assemblato esiste un percorso target per immagine.

I sampler di produzione sono
`configs/blender/domain_randomization_dataset_v1.json` e
`configs/blender/wheel_hole_anomaly_dataset_v1.json`, separati dai config
congelati dei pilot. Gli indici clean e paired sono disgiunti.

Il pool riusato e
`outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000/pool/prebulk_200_v1/`.
`manifest.jsonl` conserva riferimenti repo-relative e checksum; `prebulk.json`
registra i conteggi. Il piano bulk ha poi fissato le 150 clean nel train e
le 25 coppie nel test, senza rigenerarle o separarne i membri.

### Run e tempi del pre-bulk

| Run | Unita / RGB | Verifica | Tempo pipeline | Altre misure |
|---|---:|---|---:|---|
| `prebulk_200/clean_batch/clean_150_v1` | 150 / 150 | 150/150, 2 chunk, 150 render call | 1062,08 s (17 min 42 s) | post-processing CPU cumulativo 893,53 s |
| `prebulk_200/anomaly_batch/pairs_25_v1` | 25 coppie / 50 | 25/25, 1 chunk, 50 render call | 661,84 s (11 min 2 s) | preflight 360/360 in 145,04 s |

I percorsi sono relativi a `outputs/anomaly_detection_2/`. Tra avvio clean
e fine paired sono trascorsi 1783,05 s, cioe 29 min 43 s. La previsione di
circa 25 minuti sottostimava il totale di 4 min 43 s, pari al 18,9%. La
ricompressione e validazione PNG lato CPU, soprattutto clean, restavano il
collo di bottiglia osservato.

### Verifica del pre-bulk

Il report autorevole e
`outputs/anomaly_detection_2/prebulk_200/audit/prebulk_200_audit.json`:
esito PASS, senza errori o warning contrattuali.

- 175 unita, 200 RGB e 400 raster unici; le referenze ripetute alla target
  mask di una coppia vengono deduplicate nell'audit.
- Tutti i raster decodificabili a 1200x900; RGB in modo RGB, mask in modo L
  con soli valori 0/255, non vuote e non piene.
- Checksum di pool e artefatti validi; nessun file mancante, extra o non
  referenziato, nessun gruppo byte-identico o pixel-identico, inclusi gli RGB.
- Gli ID coincidono con i 150 record clean e 25 record paired sorgente;
  manifest ordinati e completi, validator clean/paired e tutti i gate paired
  superati, pool accettato dal validatore delle unita.
- Sorgente invariata, SHA-256
  `9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`.
- Contact sheet A/C1/C2/D ispezionate: fori visibili, coppie allineate e
  nessun caso evidente di ruota sommersa.

Le distribuzioni per immagine verificate dall'audit sono camera A/C1/C2/D
80/55/55/10 e usura current/light/evident 40/90/70. La luce e 151/49
dusty/clear anziche 150/50 per l'arrotondamento indipendente dei due run.
Lo scarto massimo e due immagini tra ruote e tre tra fasi di roll.
Questi conteggi misurati sostituiscono le precedenti descrizioni narrative
di bilanciamento esatto o scarto massimo uno.

I margini paired minimi registrati sono changed fraction 0,7429, energia
locale 0,99937, inclusione target ROI 1,0, area anomalia 260 px e lato corto
14 px. Il campione medium `dr_100198` aveva mediana delta 10/255, esattamente
sulla soglia storica: il pool restava valido, con un caso senza margine.
La successiva [policy produttiva](wheel_hole_anomaly.md#gate-fotometrici-produttivi)
separa contrasto diagnostico e requisiti strutturali.

Il [benchmark PNG](performance_and_validation.md#codifica-png) ha escluso
zlib level 1: il risparmio stimato era inferiore allo 0,4% end-to-end, con
circa il 59% di spazio aggiuntivo.

## Provenienza dei run completati

Il bulk e concluso e il dataset e assemblato. La tabella conserva la
provenienza delle vecchie note di avanzamento; config e manifest dei run
restano autorevoli. I percorsi bulk sono sotto
`outputs/anomaly_detection_2/bulk/waves/clean/` oppure `anomaly/`.

| Run | Composizione / split | RGB | Tempo registrato |
|---|---|---:|---|
| `wave_001_train_clean_1000` | 1.000 clean train | 1.000 | circa 1 h 57 min osservati |
| `wave_002_train_clean_500` | 500 clean train | 500 | 3.491,9 s, circa 58 min 12 s |
| `wave_003_train_clean_2500` | 2.500 clean train | 2.500 | 17.563,2 s, circa 4 h 52 min 43 s |
| `wave_004_train_clean_2200` | 2.200 clean train | 2.200 | 15.415,4 s, circa 4 h 16 min 55 s |
| `wave_005_clean_remaining_1150` | 650 clean train + 500 clean validation | 1.150 | non riportato nella nota originale |
| `wave_006_pairs_675_photometric_diagnostic` | 250 coppie validation + 425 coppie test | 1.350 | non riportato nella nota originale |
| `wave_007_pairs_220_test` | 220 coppie test, offset 425 | 440 | circa 77 min interni |
| `wave_008_pairs_220_test` | 220 coppie test, offset 645 | 440 | circa 77 min interni |
| `wave_009_pairs_110_test_final` | 110 coppie test, offset 865 | 220 | circa 41 min interni |

A queste 9.800 RGB si aggiungono le 200 del pre-bulk. I primi quattro
blocchi clean, completati tra il 14 e il 16 agosto 2026, hanno prodotto
6.200 clean train, con matrice camera/ruota 24/24 e validator superati.
La sequence 005, avviata il 17 agosto e descritta dal config
`configs/blender/bulk_sequences/sequence_005_2500.json`, ha eseguito in
serie i blocchi clean e paired per evitare contesa GPU. Il suo stato
`complete` non sostituiva i due validator. La stima iniziale di 7 h 15 min
(2 h 14 min clean e circa 5 h paired) era una previsione, non un tempo misurato.

Le wave paired 007/008 sono state completate il 17 agosto e la 009 il
18 agosto 2026. Non esistono sovrapposizioni tra selezioni delle wave
006--009. Dopo la sequence 005 restavano 550 coppie test, cioe 1.100
immagini, completate dalle ultime tre wave. I run diagnostici hanno registrato rispettivamente 665/10,
214/6, 214/6 e 109/1 coppie normal-contrast/low-contrast. Il validator
scansiona i raster tramite byte buffer, equivalente al precedente getpixel.
La gestione dei gate e della ripartenza e descritta nel
[contratto delle anomalie](wheel_hole_anomaly.md#gate-fotometrici-produttivi).

I tempi sopra mantengono lo scope riportato dalle note originali: wall
time osservato e tempo interno non sono intercambiabili. Le istruzioni
live di PID, monitoraggio e resume delle singole wave sono state rimosse
con l'accorpamento; la provenienza resta nei run e nei config.

## Materializzazione e sicurezza

La configurazione autorevole e
`configs/blender/final_dataset_assembly_v1.json`; l'entrypoint host e
`scripts/host/assemble_final_dataset.py`.

La procedura e fail-closed:

1. legge soltanto run completi e validati;
2. riconcilia ogni unita con il piano bulk;
3. verifica gli SHA-256 delle sorgenti;
4. costruisce la vista in staging;
5. pubblica manifest e directory soltanto dopo tutti i controlli.

Non modificare mai un PNG in place dentro la vista finale: essendo un
hardlink, la modifica altererebbe anche il raster del run sorgente. Per
trasformazioni, resize o export MVTec/Anomalib creare sempre un nuovo file in
una directory derivata. Eliminare un singolo nome hardlink non elimina gli
altri nomi dello stesso contenuto, ma la cancellazione della vista canonica va
comunque trattata come operazione distruttiva.

## Nota sui pair-lock storici

Per 1.225 coppie il contratto fotometrico produttivo e cambiato dopo la
materializzazione del piano iniziale. L'assemblatore conserva il
`pair_lock_id` effettivo e validato dal run e registra anche il valore
pianificato come `planned_pair_lock_id`. Split, sample index e categorie
semantiche restano quelli fissati dal piano; nessuna coppia viene separata.

## Pulizia post-assemblaggio

Il 18 agosto 2026 e stata eseguita una pulizia conservativa degli output. Sono
stati rimossi soltanto artefatti rigenerabili o sostituiti:

- raster dei benchmark clean e paired;
- dipendenze Python generate localmente sotto `outputs/`;
- smoke clean v1, sostituito dallo smoke v2 con gate di contatto terreno;
- revisioni visive intermedie di terreno, foro passante, bordo e profondita
  pareti, ormai incorporate nel pilot T3 canonico;
- PID terminati e log dei tre tentativi paired falliti antecedenti al run
  fotometrico diagnostico.

Sono stati conservati i run bulk e pre-bulk: oltre ai raster collegati tramite
hardlink, contengono la provenienza completa, i manifest e i metadati di
rendering. Sono stati conservati anche scena sorgente, pilot T3 canonico e
probe di drift. Dopo la pulizia sono stati ricontrollati l'hash di
`dataset.json` e i conteggi di 10.000 RGB, 10.000 target-wheel mask e 1.250
anomaly mask.

I sorgenti e le configurazioni dei benchmark rimangono nel repository: sono
piccoli e servono a riprodurre le misure documentate, anche se i raster
benchmark non vengono conservati.

## Export Kaggle

L'export minimale per Kaggle era stato creato in
`outputs/anomaly_detection_2/kaggle_export/curiosity_wheel_hole_v1_10000`.
Conteneva un unico ZIP con `images/`, `masks/` e `samples.csv`, con un README
breve esterno. L'archivio era stato verificato integralmente con 7-Zip.

Il 14 ottobre 2026 l'intera directory export e stata rimossa su richiesta,
liberando circa 11,15 GB. Il dataset canonico resta completo e conserva tutti
i 21.250 raster. Prima della rimozione sono stati estratti e conservati i
piccoli metadati dell'export in
`outputs/cleanup_20261014/kaggle_export_metadata/`.

Il `samples.csv` originale e stato conservato anche nella root del dataset
canonico: il loader ML lo richiede, mentre gli indici JSONL documentano la
generazione. Le sue 10.000 righe corrispondono esattamente a ID, split,
condizioni e percorsi dei raster in `image_manifest.jsonl`. SHA-256:
`afc4d44cf8330b94c52f21a314b797abdc553d517348b11489ac15e32a3c06fb`.
Nessun raster o indice di generazione esistente e stato modificato.

## Pulizia locale del 14 ottobre 2026

La pulizia autorizzata in questa data ha rimosso:

- gli output della pipeline legacy `blender_audit`, `wheel_preparation` e
  `rover_simulator`, insieme al relativo archivio documentale;
- copie di compilazione e QA della tesi, anteprime scartate, output di prova
  dei notebook e dell'analisi S5Mars;
- preview di domain randomization, sei backup `.blend1`, log e cache
  accessibili;
- l'ambiente Python `.venv` alla radice del repository, conservando prima un
  elenco delle 110 versioni di pacchetto installate.

Prima di rimuovere le cartelle temporanee sono stati conservati i sorgenti e
i dati dei grafici sotto `../thesis/figures/` e il registro delle fonti
dell'introduzione sotto `../thesis/research/`, rispetto alla root di questo
progetto. I riferimenti alla tesi puntano ora a `../thesis/latex/`.

Il dataset canonico, i run bulk/pre-bulk, le scene sorgente correnti, i pilot
canonici, i report `local_registry_imports` e il pilot su immagini reali non
sono stati rimossi. Lo ZIP Kaggle, inizialmente conservato, e stato rimosso
nel passaggio successivo descritto sopra. Il controllo dopo la pulizia ha
verificato la presenza di tutti i 21.250 raster e l'identita degli hash di
`dataset.json`, manifest, split e piano rispetto al controllo precedente.
I conteggi restano 10.000 RGB, 10.000 target-wheel mask e 1.250 anomaly mask,
con split 7.000/1.000/2.000.

L'elenco puntuale delle rimozioni, i sorgenti salvati e l'inventario Python
sono registrati localmente in `outputs/cleanup_20261014/`, escluso da Git.
La rimozione dei vecchi output elimina checkpoint intermedi della pipeline
precedente; i sorgenti attuali e la provenienza del dataset finale restano
conservati. Per eseguire nuovamente gli strumenti che usavano `.venv` occorre
ricreare un ambiente con le dipendenze pertinenti.
