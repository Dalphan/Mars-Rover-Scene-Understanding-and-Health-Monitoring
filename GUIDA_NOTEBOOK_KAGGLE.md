# Guida operativa ai notebook Kaggle: training FP32, QAT e PTQ

Questa guida descrive come eseguire i notebook del progetto su Kaggle. Il flusso
corretto è il seguente:

```text
training FP32 completato
├── QAT INT8 (fine-tuning con fake quantization)
└── PTQ INT8 (quantizzazione senza nuovo training)
```

QAT e PTQ sono due esperimenti indipendenti: entrambi devono partire dalla stessa
run FP32 completata, non uno dall'altro. La source run deve contenere almeno
`best.ckpt` e `run_manifest.json`.

I notebook coinvolti sono:

- training normale e QAT: `notebooks/kaggle_s5mars_segformer_training.ipynb`;
- PTQ: `notebooks/kaggle_s5mars_ptq.ipynb`.

## 1. Preparazione una tantum

### 1.1 Configurare Kaggle

1. Importare in Kaggle il notebook che si vuole eseguire.
2. Aprire le impostazioni del notebook.
3. Abilitare l'accesso a Internet, necessario per Hugging Face e Google Drive.
4. Selezionare un acceleratore NVIDIA GPU. I notebook sono pensati per le T4 di
   Kaggle e TensorRT.
5. Per una run finale usare sempre `Run All` dall'inizio, senza eseguire celle fuori
   ordine.

Il training FP32 usa fino a due GPU tramite `DataParallel`, se disponibili. QAT e
PTQ usano una sola GPU (`cuda:0`) per rendere coerenti export e benchmark.

### 1.2 Configurare il token Hugging Face

Se il dataset `Mirali33/mb-s5mars` è pubblico, il token può non essere
necessario. Per evitare problemi di autenticazione, è comunque consigliato
creare nei Kaggle Secrets entrambe queste etichette con lo stesso token Hugging
Face di sola lettura:

```text
hf_token
HF_TOKEN
```

Questa duplicazione è necessaria perché attualmente:

- il notebook di training/QAT cerca `hf_token`;
- il notebook PTQ cerca `HF_TOKEN`.

Non inserire il token direttamente nelle celle e non stamparlo negli output.

### 1.3 Configurare Google Drive (consigliato)

Google Drive è il metodo più semplice per passare una run FP32 a QAT o PTQ e
per conservare gli artefatti oltre la durata della sessione Kaggle.

Seguire la sezione **Google Drive experiment storage** del `README.md` e creare
questi Kaggle Secrets:

```text
GDRIVE_CLIENT_ID
GDRIVE_CLIENT_SECRET
GDRIVE_REFRESH_TOKEN
GDRIVE_FOLDER_ID
```

Assicurarsi che ogni secret sia collegato al notebook. La cartella associata a
`GDRIVE_FOLDER_ID` è la radice degli esperimenti, non la cartella di una singola
run.

Se non si usa Drive, i risultati restano in `/kaggle/working`. Prima di chiudere
la sessione bisogna scaricarli o salvarli come Kaggle Dataset/Notebook Output.

## 2. Training normale FP32

Notebook: `notebooks/kaggle_s5mars_segformer_training.ipynb`.

### 2.1 Configurare la cella principale

Nella cella di configurazione impostare prima di tutto:

```python
QUANTIZATION_MODE = "none"
```

Scegliere quindi il modello. Esempi:

```python
# SegFormer-B0
MODEL_NAME = "segformer_b0"

# U-Net ResNet34
MODEL_NAME = "smp"
SMP_ARCHITECTURE = "unet"
SMP_ENCODER_NAME = "resnet34"

# DeepLabV3+ MobileNetV2
MODEL_NAME = "smp"
SMP_ARCHITECTURE = "deeplabv3plus"
SMP_ENCODER_NAME = "mobilenet_v2"

# LCNet3_7
MODEL_NAME = "lcnet"
LCNET_VARIANT = "lcnet3_7"
```

Le combinazioni supportate sono:

- `segformer_b0`;
- SMP `unet`, `deeplabv3` o `deeplabv3plus`, con encoder `resnet34` o
  `mobilenet_v2`;
- `lcnet3_7` o `lcnet3_11`.

Configurare gli iperparametri della run:

```python
EPOCHS = 30
BATCH_SIZE = 32
FREEZE = "none"  # none, encoder oppure classifier
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 1

CRITERION_NAME = "combined"
CRITERION_ALPHA = 0.5
CRITERION_WEIGHT_TYPE = "square"
```

Per una run completa lasciare:

```python
LIMIT_TRAIN_BATCHES = None
LIMIT_VAL_BATCHES = None
MODEL_ANALYSIS_ENABLED = False
```

Se compare un errore CUDA out-of-memory, ridurre `BATCH_SIZE`. Non modificare
`IMAGE_SIZE`, `NUM_CLASSES` o `IGNORE_INDEX` senza voler cambiare il contratto
dell'esperimento.

### 2.2 Abilitare la persistenza su Drive

Per fare in modo che la run possa essere usata successivamente da QAT e PTQ:

```python
DRIVE_UPLOAD_ENABLED = True
DRIVE_PARENT_FOLDER_ID = None
DRIVE_UPLOAD_CHECKPOINTS = True
```

`DRIVE_UPLOAD_CHECKPOINTS` deve essere `True`: senza `best.ckpt`, QAT e PTQ non
possono usare la run remota come sorgente.

Con `DRIVE_PARENT_FOLDER_ID = None` viene usato il secret
`GDRIVE_FOLDER_ID`. La struttura remota sarà simile a:

```text
S5Mars_Experiments/
└── training/
    └── <model_run_name>/
        └── <run_id>/
```

### 2.3 Eseguire il notebook

1. Riavviare la sessione Kaggle se sono rimaste variabili da prove precedenti.
2. Eseguire tutte le celle in ordine.
3. Verificare che la cella iniziale rilevi almeno una GPU.
4. Controllare a ogni epoca `train_miou`, `val_miou` e il messaggio di salvataggio
   del checkpoint.
5. Al termine il notebook ricarica automaticamente `best.ckpt` ed esegue la
   valutazione sul test set.
6. Controllare che la cella finale riporti il percorso sotto
   `/kaggle/working/<RUN_ID>`.
7. Se Drive è abilitato, controllare che `upload_status.json` abbia stato
   `complete`.

Usare validation loss e validation mIoU per scegliere il modello e gli
iperparametri; non usare ripetutamente le metriche del test set per il tuning.

### 2.4 Artefatti da controllare

La directory della run deve contenere almeno:

```text
best.ckpt
last.ckpt
history.json
test_metrics.json
prediction_grid.png
run_manifest.json
environment.json
artifacts_manifest.json
upload_status.json
```

Conservare per ogni run:

- `RUN_ID`;
- `model_run_name` nel `run_manifest.json`;
- SHA-256 di `best.ckpt` nel manifest;
- `drive_folder_id` presente in `upload_status.json`.

Quest'ultimo è l'ID da usare come source folder per QAT o PTQ.

## 3. Quantization-Aware Training (QAT)

Notebook: `notebooks/kaggle_s5mars_segformer_training.ipynb`, lo stesso del
training FP32.

### 3.1 Prerequisiti della source run

Prima di iniziare devono esistere:

```text
best.ckpt
run_manifest.json
```

provenienti da un training FP32 completato. Il modello configurato nel notebook
QAT deve essere esattamente lo stesso della source run. Per esempio, una source
`smp_unet_resnet34` richiede nuovamente:

```python
MODEL_NAME = "smp"
SMP_ARCHITECTURE = "unet"
SMP_ENCODER_NAME = "resnet34"
```

Il notebook verifica `model_run_name`, schema del manifest e SHA-256 del
checkpoint; una configurazione non corrispondente viene rifiutata.

### 3.2 Usare una source run da Google Drive

Nella cella di configurazione impostare:

```python
QUANTIZATION_MODE = "qat_int8"
QAT_SOURCE_MODE = "drive"
QAT_SOURCE_DRIVE_FOLDER_ID = "ID_CARTELLA_DEL_RUN_FP32"
QAT_SOURCE_RUN_DIR = None
```

`QAT_SOURCE_DRIVE_FOLDER_ID` deve essere l'ID della cartella precisa
`training/<model>/<run_id>`, non `GDRIVE_FOLDER_ID`.

### 3.3 Alternativa: source run locale/Kaggle Input

Allegare la directory della run FP32 al notebook come Kaggle Input e impostare:

```python
QUANTIZATION_MODE = "qat_int8"
QAT_SOURCE_MODE = "local"
QAT_SOURCE_RUN_DIR = "/kaggle/input/<dataset>/<cartella-run>"
QAT_SOURCE_DRIVE_FOLDER_ID = None
```

`QAT_SOURCE_RUN_DIR` deve indicare direttamente la directory che contiene
`best.ckpt` e `run_manifest.json`.

### 3.4 Configurare il QAT

Valori iniziali consigliati, già presenti nel notebook:

```python
QAT_CALIBRATION_SIZE = 128
QAT_CALIBRATION_SEED = 42
QAT_CALIBRATION_BATCH_SIZE = 1

QAT_EPOCHS = 5
QAT_LR = 1e-5
QAT_WEIGHT_DECAY = 0.01
QAT_FREEZE = "none"

QAT_EXPORT_TENSORRT = True
QAT_BENCHMARK_WARMUP_ITERATIONS = 100
QAT_BENCHMARK_MEASUREMENT_ITERATIONS = 1000
QAT_BENCHMARK_TRIALS = 3
```

Quando `QUANTIZATION_MODE = "qat_int8"`, i normali `EPOCHS`, `LR`,
`WEIGHT_DECAY` e `FREEZE` vengono sovrascritti automaticamente dai corrispondenti
parametri `QAT_*`.

Per caricare anche la run QAT su Drive:

```python
DRIVE_UPLOAD_ENABLED = True
DRIVE_PARENT_FOLDER_ID = None
DRIVE_UPLOAD_CHECKPOINTS = True
```

### 3.5 Eseguire e verificare il QAT

1. Avviare una sessione Kaggle pulita.
2. Eseguire tutte le celle in ordine.
3. Verificare il download e la validazione della source run.
4. Controllare che il modello FP32 venga caricato prima dell'applicazione della
   quantizzazione; il QAT non deve partire da pesi casuali.
5. Controllare le metriche train/validation durante le epoche QAT.
6. Verificare che venga selezionato il miglior checkpoint tramite validation
   mIoU.
7. Lasciare eseguire la cella di export Q/DQ e costruzione TensorRT.
8. Se la build TensorRT fallisce, leggere
   `logs/build_qat_int8.log` prima di modificare il notebook.
9. Controllare `upload_status.json` se Drive è abilitato.

Artefatti principali attesi:

```text
best.ckpt
last.ckpt
best_modelopt.pth
last_modelopt.pth
history.json
test_metrics.json
model_int8_qat_qdq.onnx
engine_int8_qat.plan
tensorrt_test_metrics.json
quantization_summary.json
benchmark_results.json
run_manifest.json
artifacts_manifest.json
upload_status.json
logs/build_qat_int8.log
```

`test_metrics.json` descrive il modello PyTorch con fake quantization;
`tensorrt_test_metrics.json` descrive il vero engine TensorRT. Non confrontare le
due misure come se fossero lo stesso runtime.

## 4. Post-Training Quantization (PTQ)

Notebook: `notebooks/kaggle_s5mars_ptq.ipynb`.

Il PTQ non esegue training. Carica il `best.ckpt` FP32, esporta ONNX, produce gli
engine TensorRT FP32/FP16/INT8, misura le metriche e confronta le latenze.

### 4.1 Configurare la source run da Drive

```python
SOURCE_MODE = "drive"
SOURCE_DRIVE_FOLDER_ID = "ID_CARTELLA_DEL_RUN_FP32"
SOURCE_RUN_DIR = None
```

Anche qui serve l'ID della cartella precisa del run FP32.

### 4.2 Alternativa: source run locale/Kaggle Input

```python
SOURCE_MODE = "local"
SOURCE_RUN_DIR = "/kaggle/input/<dataset>/<cartella-run>"
SOURCE_DRIVE_FOLDER_ID = None
```

La directory deve contenere direttamente `best.ckpt` e `run_manifest.json`.

### 4.3 Configurare precisioni, calibrazione e benchmark

Per il confronto completo:

```python
RUN_FP32_TRT = True
RUN_FP16_TRT = True
RUN_INT8_PTQ = True

CALIBRATION_SIZE = 128
CALIBRATION_SEED = 42
CALIBRATION_STRATEGY = "random"  # oppure class_coverage
CALIBRATION_METHOD = "entropy"  # oppure max
CALIBRATION_BATCH_SIZE = 1

BENCHMARK_BATCH_SIZE = 1
BENCHMARK_WARMUP_ITERATIONS = 100
BENCHMARK_MEASUREMENT_ITERATIONS = 1000
BENCHMARK_TRIALS = 3
```

`CALIBRATION_BATCH_SIZE` e `BENCHMARK_BATCH_SIZE` devono rimanere uguali a 1,
perché input ONNX ed engine sono statici con shape `1 x 3 x 512 x 512`.

Durante la scelta della configurazione lasciare:

```python
RUN_FINAL_TEST = False
```

Confrontare `random`/`class_coverage` e `entropy`/`max` usando soltanto il train
per calibrazione e la validation per selezione. Dopo aver congelato la
configurazione finale, eseguire una nuova run pulita con:

```python
RUN_FINAL_TEST = True
```

Il test set deve essere usato una sola volta per il risultato finale.

### 4.4 Eseguire prima uno smoke test

Prima di una run completa è consigliato verificare l'intera pipeline con:

```python
CALIBRATION_SIZE = 4
LIMIT_VAL_BATCHES = 2
LIMIT_TEST_BATCHES = 2
BENCHMARK_WARMUP_ITERATIONS = 5
BENCHMARK_MEASUREMENT_ITERATIONS = 10
BENCHMARK_TRIALS = 1
RUN_FINAL_TEST = False
DRIVE_UPLOAD_ENABLED = False
```

Lo smoke test deve completare almeno export ONNX, quantizzazione INT8, build dei
tre engine e benchmark. I suoi risultati non vanno usati nella tesi.

### 4.5 Eseguire la run PTQ completa

1. Avviare una sessione Kaggle pulita con GPU NVIDIA e Internet abilitato.
2. Eseguire le celle in ordine; la prima installa le dipendenze mancanti.
3. Controllare che `DEVICE` sia `cuda:0` e che TensorRT sia rilevato.
4. Verificare caricamento, schema e SHA-256 della source run.
5. Attendere l'export `model_fp32.onnx` e il controllo ONNX Runtime. Il notebook
   richiede pixel agreement almeno `0.99`.
6. Attendere la calibrazione INT8, che usa esclusivamente il train split.
7. Attendere la costruzione degli engine TensorRT.
8. Controllare i log sotto `logs/` in caso di errore.
9. Verificare metriche di validation e `benchmark_results.json`.
10. Per la sola configurazione finale, verificare anche le metriche test.
11. Controllare `upload_status.json` se l'upload Drive è abilitato.

Per caricare la run PTQ:

```python
DRIVE_UPLOAD_ENABLED = True
DRIVE_PARENT_FOLDER_ID = None
```

Artefatti principali attesi, in funzione dei flag abilitati:

```text
model_fp32.onnx
model_fp16.onnx
model_int8_qdq.onnx
engine_fp32.plan
engine_fp16.plan
engine_int8.plan
export_validation.json
quantization_summary.json
engine_build_results.json
val_metrics_pytorch_fp32.json
val_metrics_fp32.json
val_metrics_fp16.json
val_metrics_int8.json
test_metrics_*.json           # solo con RUN_FINAL_TEST=True
benchmark_results.json
comparison_metrics.json
calibration_manifest.json
run_manifest.json
artifacts_manifest.json
upload_status.json
logs/
```

## 5. Come trovare l'ID corretto della source run

Non usare direttamente `GDRIVE_FOLDER_ID`: quello identifica soltanto
`S5Mars_Experiments`.

Usare uno di questi metodi:

1. aprire l'`upload_status.json` della run FP32 e copiare `drive_folder_id`;
2. aprire la cartella specifica del run in Google Drive e copiare la parte finale
   dell'URL:

```text
https://drive.google.com/drive/folders/ID_DA_COPIARE
```

La struttura prevista è:

```text
S5Mars_Experiments/
├── training/<model_run_name>/<run_id>/
├── qat/<model_run_name>/<source_run_id>/<run_id>/
└── ptq/<model_run_name>/<source_run_id>/<run_id>/
```

Per `QAT_SOURCE_DRIVE_FOLDER_ID` e `SOURCE_DRIVE_FOLDER_ID` usare sempre la
cartella sotto `training/`.

## 6. Sequenza consigliata per ogni modello

1. Eseguire il training FP32 completo.
2. Controllare `best.ckpt`, `run_manifest.json` e stato upload.
3. Annotare `run_id`, mIoU di validation/test e Drive folder ID.
4. Eseguire QAT partendo da quella run FP32.
5. Eseguire PTQ partendo dalla stessa run FP32.
6. Confrontare i risultati mantenendo costanti GPU, preprocessing, batch size e
   protocollo di benchmark.
7. Archiviare insieme manifest, metriche, benchmark e log di build.

Non usare una run QAT come sorgente del notebook PTQ.

## 7. Problemi comuni

### Secret Hugging Face non trovato

Verificare il nome richiesto dal notebook:

- training/QAT: `hf_token`;
- PTQ: `HF_TOKEN`.

### `Missing required Kaggle secret: GDRIVE_*`

Il secret manca oppure non è collegato al notebook. Controllare tutti e quattro
i secret Drive nel pannello Kaggle.

### `invalid_grant` da Google OAuth

Il refresh token può essere scaduto o revocato. Ripetere localmente
`scripts/setup_google_drive_oauth.py` e aggiornare i Kaggle Secrets.

### `Set SOURCE_DRIVE_FOLDER_ID` o `Set QAT_SOURCE_DRIVE_FOLDER_ID`

È stato lasciato `None`, oppure è stato indicato `GDRIVE_FOLDER_ID` invece
della cartella esatta della run FP32.

### `model mismatch`

La configurazione modello del QAT non coincide con `model_run_name` della source
run. Riallineare `MODEL_NAME` e i parametri SMP/LCNet.

### Errore SHA-256

Il checkpoint non corrisponde al manifest, è incompleto oppure proviene da
un'altra run. Non disabilitare il controllo: recuperare nuovamente entrambi i
file dalla stessa cartella.

### Errore TensorRT

Leggere prima il file sotto `logs/`. Verificare GPU NVIDIA, disponibilità di
TensorRT/trtexec e spazio libero in `/kaggle/working`.

### CUDA out-of-memory

Nel training FP32/QAT ridurre `BATCH_SIZE`. Nel PTQ non aumentare i batch size
statici oltre 1.
