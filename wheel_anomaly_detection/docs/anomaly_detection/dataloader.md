# Dataloader minimale

Stato: implementato.

## Contratto

`CuriosityWheelDataset` legge `samples.csv` dalla directory estratta da Kaggle
e rende disponibili gli split `train`, `validation` e `test`.

Ogni campione restituisce:

- `image`: tensore RGB `float32`; senza normalizzazione i valori sono in
  `[0, 1]`;
- `target_mask`: tensore `uint8` della ruota target;
- `anomaly_mask`: tensore `uint8`, tutto zero per i clean;
- `label`: `0` clean, `1` hole;
- `has_anomaly_mask` e i metadati di `samples.csv`.

Il builder riceve una configurazione Hydra e restituisce tre `DataLoader`
nell'ordine train, validation e test: train usa uno shuffle deterministico,
validation e test non usano shuffle.

## Preprocessing opzionale

`PreprocessingConfig` lascia ogni passaggio disabilitato con `None` e supporta:

- resize `(height, width)`, applicato anche alle mask;
- normalizzazione RGB tramite mean/std;
- jitter di luminosita, contrasto, gamma e saturazione;
- rumore sensore dipendente dal segnale;
- Gaussian noise additivo e Gaussian blur con intervallo di sigma.

Il resize usa bilinear antialias per RGB e nearest-neighbor per le mask. Le
trasformazioni fotometriche non modificano le mask. Il builder accetta
`train_preprocessing` ed `evaluation_preprocessing` separati, per evitare di
applicare augmentation stocastica a validation e test.

Dimensione, media, deviazione standard e abilitazione delle augmentation train
sono proprietà della configurazione del modello. Il default PatchCore esegue un
resize diretto a `256x256`, usa la normalizzazione ImageNet e disabilita le
augmentation casuali per non contaminare la distribuzione della memory bank.

La conversione RGB a `float32` in `[0, 1]` e sempre applicata, così train,
validation e test condividono lo stesso contratto numerico. La normalizzazione,
se configurata, viene applicata dopo le augmentation. `audit_preprocessing`
mostra immagini originali e varianti trasformate con un seed locale
riproducibile.

Il manifest viene validato integralmente prima di selezionare lo split:
split e condizioni devono essere noti, gli `image_id` univoci, ogni coppia deve
contenere esattamente un clean e un hole nello stesso split e tutti gli
artefatti dello split richiesto devono esistere.

## Uso da terminale

Dalla directory `wheel_anomaly_detection`:

```powershell
python scripts/anomaly_detection/inspect_dataloaders.py `
  dataset.root="D:/datasets/curiosity_wheel_hole_v1_10000"
```

I default di DataLoader e preprocessing sono in
`configs/anomaly_detection/config.yaml` e possono essere sovrascritti da
CLI, per esempio con `dataloader.batch_size=8`.

Su Kaggle, `<dataset-root>` è la directory sotto `/kaggle/input` che contiene
direttamente `samples.csv`, `images/` e `masks/`.

Il notebook Kaggle cerca automaticamente questa directory ed è autosufficiente:
duplica il codice minimo del DataLoader invece di importare i moduli del
progetto. Questo semplifica l'esecuzione, ma richiede di mantenere sincronizzate
le due versioni. Come nel notebook della segmentazione semantica, tutti i
parametri modificabili sono raccolti nella cella iniziale `Configuration`.

## Configurazione corrente del notebook

- Il resize diretto a `256x256` è un baseline temporaneo e configurabile.
- Train, validation e test usano la normalizzazione ImageNet. È coerente con i
  backbone preaddestrati di PatchCore/PaDiM e con il preprocessing di
  EfficientAD.
- PatchCore disabilita le augmentation train; altri modelli potranno abilitarle
  dalla propria configurazione.
- L'audit visuale mostra il preprocessing effettivamente scelto.

## Trade-off della milestone

- Non vengono ancora applicati crop o ROI.
- Il resize quadrato deforma il rapporto nativo `4:3`; consente però un baseline
  semplice, riproducibile e confrontabile prima dello studio della risoluzione.
- Le anomaly mask mancanti dei clean diventano maschere zero per mantenere uno
  schema di batch uniforme.
- Le immagini hanno dimensioni uniformi nel dataset corrente; il collate
  standard fallirà esplicitamente se in futuro verranno mescolate risoluzioni
  diverse.
- La scansione iniziale degli artefatti aggiunge operazioni di metadata I/O,
  ma evita che file mancanti emergano soltanto a training già avviato.
