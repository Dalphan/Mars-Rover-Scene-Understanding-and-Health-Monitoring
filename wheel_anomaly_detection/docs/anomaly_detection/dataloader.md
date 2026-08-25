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

- resize diretto `(height, width)` oppure resize del lato corto;
- center crop opzionale, applicato in modo identico a RGB e mask;
- normalizzazione RGB tramite mean/std;
- jitter di luminosita, contrasto, gamma e saturazione;
- rumore sensore dipendente dal segnale;
- Gaussian noise additivo e Gaussian blur con intervallo di sigma.

Il resize usa bilinear antialias per RGB e nearest-neighbor per le mask. Le
trasformazioni fotometriche non modificano le mask. Il builder accetta
`train_preprocessing` ed `evaluation_preprocessing` separati, per evitare di
applicare augmentation stocastica a validation e test.

Dimensione, geometria e normalizzazione appartengono al preset del modello.
`model=patchcore_light` usa un resize diretto a `384x512`, preservando il
rapporto nativo `4:3`; `model=patchcore_reference` replica la geometria Amazon
con resize del lato corto a `256` e center crop `224x224`. Entrambi usano la
normalizzazione ImageNet e disabilitano le augmentation train.

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

- `PATCHCORE_PRESET="light"` usa `384x512`; `"reference"` usa
  `Resize(256) + CenterCrop(224)`.
- Train, validation e test usano la normalizzazione ImageNet. È coerente con i
  backbone preaddestrati di PatchCore/PaDiM e con il preprocessing di
  EfficientAD.
- PatchCore disabilita le augmentation train; altri modelli potranno abilitarle
  dalla propria configurazione.
- L'audit visuale mostra il preprocessing effettivamente scelto.

## Trade-off della milestone

- Il center crop è applicato solo nel preset reference per confrontabilità con
  Amazon; il preset light conserva tutto il frame `4:3`.
- Una ROI basata sulla target mask non viene usata, perché tale mask potrebbe
  non essere disponibile al deployment.
- Le anomaly mask mancanti dei clean diventano maschere zero per mantenere uno
  schema di batch uniforme.
- Le immagini hanno dimensioni uniformi nel dataset corrente; il collate
  standard fallirà esplicitamente se in futuro verranno mescolate risoluzioni
  diverse.
- La scansione iniziale degli artefatti aggiunge operazioni di metadata I/O,
  ma evita che file mancanti emergano soltanto a training già avviato.
