# Piano di implementazione PTQ e QAT

## 1. Scopo

Questo documento definisce l'implementazione della fase di quantizzazione dei
modelli di segmentazione S5Mars. La fase copre:

- inferenza TensorRT FP32 usata come baseline del runtime;
- inferenza TensorRT FP16;
- Post-Training Quantization INT8;
- Quantization-Aware Training INT8;
- valutazione di accuratezza con le metriche già esistenti;
- benchmark comparativo su una singola NVIDIA T4 con batch size 1;
- persistenza locale e upload opzionale su una cartella Google Drive;
- artefatti sufficienti a riprodurre e analizzare ogni run.

La NVIDIA T4 è una piattaforma proxy comune a tutte le run. L'obiettivo non è
dimostrare un requisito assoluto di real-time sul rover, ma confrontare modelli
e precisioni in condizioni controllate. I risultati non devono essere
presentati come misure dirette di un futuro hardware edge o space-grade.

## 2. Decisioni già approvate

1. Input di inferenza fisso: batch 1, RGB, 512 x 512.
2. Hardware del benchmark: una sola NVIDIA T4.
3. Nessuna soglia minima di FPS.
4. Le misure principali sono differenze di mIoU, latenza, speedup, memoria e
   dimensione degli artefatti.
5. FP16 è una modalità di inferenza a precisione ridotta e non richiede
   calibrazione.
6. INT8 PTQ usa un calibration subset ricavato esclusivamente dal train split.
7. Validation serve a scegliere configurazione PTQ, checkpoint e fallback.
8. Test serve alla valutazione finale delle configurazioni congelate.
9. PTQ viene implementato in un nuovo notebook Kaggle autosufficiente.
10. QAT viene integrato nel notebook di training esistente tramite una flag.
11. I file history.json e test_metrics.json e le relative metriche restano
    invariati. Non rientrano nello scope modifiche specifiche alla classe 0.
12. Metadati, benchmark e risultati di quantizzazione sono salvati in file
    aggiuntivi e separati.
13. Ogni run conserva best.ckpt: le sole metriche non sono sufficienti per
    applicare PTQ o riprodurre QAT.
14. Gli artefatti possono essere caricati in una cartella My Drive posseduta
    dall'account principale e condivisa con un secondo account.

## 3. Domande di ricerca supportate

- RQ1: come cambia il trade-off accuratezza-latenza passando da TensorRT FP32
  a FP16 e INT8 sulla stessa T4?
- RQ2: la risposta alla quantizzazione dipende dalla famiglia architetturale
  del modello?
- RQ3: dimensione e composizione del calibration set cambiano in modo
  significativo il risultato INT8?
- RQ4: quando PTQ INT8 degrada l'accuratezza, QAT recupera qualità mantenendo
  il vantaggio di inferenza?
- RQ5: un modello più accurato quantizzato domina un modello nativamente
  leggero, oppure il modello leggero resta sul fronte di Pareto?

## 4. Confini dello scope

### Incluso

- tutti i modelli già costruibili dal notebook: SegFormer-B0, varianti SMP e
  LCNet;
- export ONNX con shape di input fissa;
- Q/DQ esplicito per INT8;
- TensorRT FP32, FP16 e INT8;
- calibration set deterministico;
- QAT INT8 a partire da un checkpoint FP32;
- benchmark model-only e trasferimenti separati;
- upload e recupero degli artefatti tramite Google Drive API.

### Escluso dal core

- INT4, FP8 e formati Blackwell-specific;
- pruning, distillazione e riduzione della risoluzione;
- benchmark CPU o OpenVINO;
- dichiarazioni di prestazione sul rover basate solo sulla T4;
- modifica delle formule o delle chiavi delle metriche correnti;
- uso del test set per scegliere calibration method o iperparametri QAT.

Queste esclusioni possono diventare lavori futuri, ma non devono essere
mescolate al core PTQ/QAT perché renderebbero difficile attribuire le cause dei
risultati.

## 5. Architettura generale

Il flusso implementato sarà:

    Training FP32
        |
        +-- best.ckpt + configurazione completa + hash
        |
        +-- PTQ notebook
        |     +-- PyTorch FP32 reference
        |     +-- ONNX FP32
        |     +-- TensorRT FP32
        |     +-- TensorRT FP16
        |     +-- INT8 PTQ Q/DQ
        |     +-- TensorRT INT8
        |
        +-- Training notebook in modalità QAT
              +-- caricamento best.ckpt FP32
              +-- inizializzazione fake quantization
              +-- fine-tuning QAT
              +-- checkpoint QAT
              +-- ONNX Q/DQ
              +-- TensorRT INT8 QAT

I due rami devono usare lo stesso preprocessing, lo stesso test loader, lo
stesso postprocessing e lo stesso protocollo di benchmark.

## 6. Prerequisiti comuni

### 6.1 Identificazione univoca della run

Sostituire le directory basate soltanto sul nome del modello con un RUN_ID
univoco. Formato consigliato:

    <model_run_name>__seed<seed>__<YYYYMMDD_HHMMSS>

Non sovrascrivere una directory esistente. Ogni run deve dichiarare:

- run_id;
- source_run_id, quando deriva da una run FP32;
- seed;
- modello e variante;
- timestamp UTC;
- tipo di run: fp32_training, ptq oppure qat_int8.

### 6.2 Configurazione ricostruibile

Il checkpoint e run_manifest.json devono includere tutti i dati necessari a
ricostruire il modello senza modificare manualmente celle:

- famiglia e nome del modello;
- checkpoint pretrained originale, se presente;
- architettura ed encoder SMP;
- tutti i parametri LCNet;
- num_classes e canali di input;
- image size, mean e standard deviation;
- seed;
- freeze mode;
- loss e iperparametri di training;
- split del dataset e repo_id;
- versione/schema del manifest.

L'estensione del manifest o del checkpoint non deve cambiare history.json o
test_metrics.json.

### 6.3 Best checkpoint

- Salvare best.ckpt e last.ckpt come oggi.
- La valutazione finale e ogni quantizzazione devono partire da best.ckpt.
- Calcolare SHA-256 di best.ckpt dopo la chiusura del file.
- Salvare hash, dimensione, epoca e best validation mIoU nel manifest.
- PTQ e QAT devono verificare l'hash dopo un download da Drive.

### 6.4 Environment probe

All'inizio di ogni notebook raccogliere e salvare, senza segreti:

- modello e quantità di GPU visibili;
- compute capability;
- driver NVIDIA;
- CUDA e cuDNN;
- Python e PyTorch;
- ONNX e ONNX Runtime;
- TensorRT e trtexec;
- NVIDIA Model Optimizer;
- segmentation-models-pytorch e transformers;
- output di nvidia-smi sintetico;
- elenco delle dipendenze installate rilevanti.

Il benchmark deve usare una sola T4 anche quando Kaggle espone due GPU.
L'eventuale DataParallel resta una scelta del training FP32, non del benchmark
né del percorso QAT iniziale.

### 6.5 Versioni delle dipendenze

Non fissare nel piano numeri di versione non ancora verificati sull'immagine
Kaggle. L'implementazione deve:

1. rilevare le versioni preinstallate;
2. installare solo i pacchetti mancanti o incompatibili;
3. eseguire uno smoke test di export e build;
4. salvare le versioni funzionanti nel manifest;
5. dopo lo smoke test, documentare la combinazione riproducibile nel notebook.

La pipeline primaria usa NVIDIA Model Optimizer per fake quantization/PTQ e
Q/DQ esplicito, ONNX come formato di scambio e TensorRT come runtime.

## 7. Notebook PTQ dedicato

### 7.1 File

Creare:

    notebooks/kaggle_s5mars_ptq.ipynb

Il notebook deve essere autosufficiente come quello di training e non deve
dipendere da import dal package src durante l'esecuzione Kaggle.

### 7.2 Configurazione utente

La prima cella di configurazione deve esporre almeno:

    SOURCE_MODE = "drive"             # drive oppure local
    SOURCE_RUN_DIR = None
    SOURCE_DRIVE_FOLDER_ID = None
    OUTPUT_ROOT = "/kaggle/working"

    RUN_FP32_TRT = True
    RUN_FP16_TRT = True
    RUN_INT8_PTQ = True

    CALIBRATION_SIZE = 128
    CALIBRATION_SEED = 42
    CALIBRATION_STRATEGY = "random"
    CALIBRATION_BATCH_SIZE = 1

    BENCHMARK_BATCH_SIZE = 1
    BENCHMARK_WARMUP_ITERATIONS = 100
    BENCHMARK_MEASUREMENT_ITERATIONS = 1000

    DRIVE_UPLOAD_ENABLED = False
    DRIVE_PARENT_FOLDER_ID = None
    DRIVE_UPLOAD_CHECKPOINTS = True

Le celle successive non devono richiedere modifiche per cambiare modello.

### 7.3 Caricamento della source run

1. Recuperare best.ckpt, run_manifest.json e configurazione.
2. Verificare hash e schema.
3. Ricostruire il modello corretto.
4. Caricare i pesi in FP32.
5. Disabilitare DataParallel.
6. Portare il modello in eval mode.
7. Eseguire uno smoke forward con shape 1 x 3 x 512 x 512.
8. Verificare output 1 x 9 x 512 x 512 e valori finiti.
9. Fallire con messaggio chiaro se un campo di configurazione è assente.

### 7.4 Data loader

Costruire loader separati:

- calibration loader: subset deterministico del train;
- validation loader: scelta configurazioni PTQ;
- test loader: valutazione finale;
- benchmark loader: campioni reali e preprocessing identico.

Salvare calibration_manifest.json con:

- repo e split;
- seed;
- strategia;
- numero richiesto ed effettivo di immagini;
- indici delle immagini;
- batch size;
- trasformazioni;
- hash della lista degli indici.

Test e validation non devono mai entrare nella calibrazione.

### 7.5 Export ONNX FP32

Usare input e output con nomi stabili e shape fissa:

    input:  [1, 3, 512, 512]
    logits: [1, 9, 512, 512]

Passi obbligatori:

1. export in eval e inference mode;
2. ONNX checker;
3. shape inference;
4. controllo di operatori non supportati;
5. inferenza ONNX Runtime su un insieme deterministico di immagini;
6. confronto con PyTorch FP32:
   - errore assoluto medio e massimo dei logits;
   - cosine similarity;
   - pixel agreement dopo argmax;
   - metriche esistenti sul validation subset;
7. salvataggio di export_validation.json.

Se l'export FP32 non supera i controlli, non procedere a FP16 o INT8.

### 7.6 TensorRT FP32

Costruire prima un engine TensorRT FP32. Serve a isolare:

- variazioni dovute all'export;
- fusioni e tactic selection;
- differenze tra PyTorch e runtime;
- baseline corretta per speedup FP16 e INT8.

Salvare:

- engine_fp32.plan;
- log di build;
- tactic/timing cache quando disponibile;
- tempo di build;
- dimensione dell'engine;
- precision coverage;
- validation e test metrics con lo stesso schema di test_metrics.json.

### 7.7 TensorRT FP16

FP16 non deve utilizzare calibration data.

1. Convertire o esportare il grafo nella modalità supportata dalla versione
   TensorRT rilevata.
2. Mantenere in precisione maggiore gli operatori che il runtime non può
   eseguire correttamente in FP16.
3. Salvare precision coverage e layer information.
4. Valutare accuratezza e benchmark con lo stesso protocollo FP32.
5. Salvare engine_fp16.plan e test_metrics_fp16.json.

### 7.8 PTQ INT8

Configurazione core:

- pesi INT8 per-channel quando supportato;
- attivazioni INT8 per-tensor;
- quantizzazione statica W8A8;
- calibrazione deterministica;
- ONNX con QuantizeLinear/DequantizeLinear espliciti;
- fallback FP16/FP32 soltanto per operatori non quantizzabili.

Passi:

1. inserire quantizer/observer secondo la configurazione;
2. eseguire il calibration loader senza gradienti;
3. esportare model_int8_qdq.onnx;
4. verificare presenza e posizione dei nodi Q/DQ;
5. produrre un report di layer quantizzati e non quantizzati;
6. costruire engine_int8.plan;
7. valutare validation;
8. congelare la configurazione;
9. valutare test;
10. eseguire benchmark batch 1;
11. salvare test_metrics_int8.json e quantization_summary.json.

Il risultato non deve essere chiamato INT8 completo se porzioni rilevanti
rimangono FP16/FP32. La precision coverage deve essere parte dei risultati.

### 7.9 Recovery path PTQ

Il core si ferma alla configurazione INT8 standard. Se la perdita rispetto a
TensorRT FP32 supera la soglia preregistrata, provare in ordine:

1. controllare rappresentatività del calibration set;
2. aumentare calibration size;
3. strategia class-coverage;
4. calibration algorithm alternativo supportato;
5. mantenere in FP16 gli operatori/layer più sensibili;
6. applicare SmoothQuant solo quando giustificato, in particolare per blocchi
   transformer;
7. passare al QAT.

Ogni recovery deve essere una run distinta e non sovrascrivere il core.

## 8. Matrice PTQ

### 8.1 Core obbligatorio

Per ogni checkpoint finale:

| Variante | Calibrazione | Scopo |
|---|---:|---|
| PyTorch FP32 | no | riferimento numerico |
| TensorRT FP32 | no | baseline runtime |
| TensorRT FP16 | no | reduced precision |
| TensorRT INT8 PTQ | 128 train images | PTQ principale |

### 8.2 Ablation calibration

Non moltiplicare subito tutte le combinazioni. Procedere a livelli:

Fase A, tutti i modelli:

- random, seed 42, 128 immagini.

Fase B, modelli rappresentativi o problematici:

- random 32;
- random 512;
- class-coverage 128.

Fase C, solo se necessario:

- algoritmo di calibrazione alternativo;
- selective/mixed precision;
- SmoothQuant per SegFormer.

Per le strategie casuali usare gli stessi indici tra configurazioni compatibili
quando la dimensione lo consente: i primi 32 devono essere un subset dei 128 e
i primi 128 un subset dei 512. Questo riduce una fonte di variabilità.

### 8.3 Ripetizioni

- Le conversioni FP32 e FP16 non richiedono seed di calibrazione.
- Il core INT8 usa seed 42.
- Per stimare sensibilità alla selezione dei campioni, ripetere almeno la
  configurazione INT8 principale con seed 42, 123 e 999 sui modelli scelti per
  l'ablation.
- Il benchmark di ogni engine deve avere almeno tre trial indipendenti nella
  stessa sessione Kaggle.

## 9. QAT nel notebook di training

### 9.1 Flag

Nel notebook esistente aggiungere:

    QUANTIZATION_MODE = "none"        # none oppure qat_int8

Una enum stringa è preferibile a un booleano perché rende gli errori di
configurazione espliciti e permette estensioni future senza cambiare
interfaccia.

### 9.2 Configurazione QAT

Quando QUANTIZATION_MODE vale qat_int8, richiedere:

    QAT_SOURCE_MODE = "drive"
    QAT_SOURCE_RUN_DIR = None
    QAT_SOURCE_DRIVE_FOLDER_ID = None
    QAT_CALIBRATION_SIZE = 128
    QAT_CALIBRATION_SEED = 42
    QAT_EPOCHS = 5
    QAT_LR = 1e-5
    QAT_WEIGHT_DECAY = <valore esplicito>
    QAT_FREEZE = "none"
    QAT_EXPORT_TENSORRT = True

I valori sono default iniziali, non risultati scientifici. Devono restare
configurabili e finire nel manifest.

### 9.3 Precondizioni

QAT deve:

1. richiedere un best.ckpt FP32 esistente;
2. verificare hash e compatibilità del modello;
3. ricostruire il modello in FP32;
4. caricare i pesi prima di inserire fake quantization;
5. inizializzare scale/observer con il train calibration subset;
6. ricreare optimizer e scheduler dopo la trasformazione QAT;
7. usare una sola T4 nella prima implementazione;
8. non ripristinare optimizer/scheduler della run FP32;
9. collegare source_run_id e source_checkpoint_sha256.

Non supportare QAT from scratch nel core, perché renderebbe il confronto con
PTQ non controllato.

### 9.4 Training QAT

- Riutilizzare train e validation loop esistenti.
- Mantenere history.json e test_metrics.json invariati.
- Usare validation mIoU per selezionare best.ckpt QAT.
- Salvare last.ckpt e best.ckpt QAT in una directory separata.
- Salvare anche lo stato specifico di NVIDIA Model Optimizer necessario a
  ricostruire quantizer e fake-quant modules.
- Non affidarsi al solo state_dict standard se perde metadati di
  quantizzazione.
- Registrare learning rate, numero di epoche, freeze mode e calibration
  manifest.

### 9.5 Export e benchmark QAT

Solo in modalità qat_int8, aggiungere dopo il training:

1. caricamento del best QAT checkpoint;
2. export ONNX Q/DQ;
3. verifica del grafo;
4. build TensorRT INT8;
5. valutazione con le metriche esistenti;
6. benchmark identico a PTQ;
7. salvataggio di:
   - model_int8_qat_qdq.onnx;
   - engine_int8_qat.plan;
   - quantization_summary.json;
   - benchmark_results.json;
   - export_validation.json.

La valutazione fake-quantized non sostituisce la valutazione del vero engine
TensorRT.

### 9.6 Regressione modalità standard

Con QUANTIZATION_MODE uguale a none:

- il notebook deve seguire il percorso FP32 corrente;
- history.json e test_metrics.json devono conservare chiavi e significato;
- non devono essere creati quantizer;
- non devono essere richieste dipendenze QAT durante il training se la sezione
  non è attivata, salvo installazione comune motivata;
- a seed e ambiente uguali, la nuova struttura non deve cambiare i risultati
  del percorso standard oltre alla normale non-deterministicità GPU.

## 10. Protocollo di accuratezza

Per ogni variante:

- usare lo stesso checkpoint sorgente;
- usare gli stessi split;
- usare lo stesso resize e normalizzazione;
- usare la stessa funzione di valutazione già presente;
- salvare le stesse chiavi di test_metrics.json:
  loss, pixel_accuracy, miou e per_class_iou;
- salvare differenze da TensorRT FP32 in un file separato;
- non modificare le metriche per la classe 0 in questa fase.

File aggiuntivo comparison_metrics.json:

- delta_miou assoluto;
- delta_pixel_accuracy;
- delta_per_class_iou;
- pixel agreement con TensorRT FP32, se calcolato;
- source precision e target precision;
- numero di campioni valutati.

Una soglia iniziale configurabile per avviare il recovery path può essere
MAX_ALLOWED_MIOU_DROP = 0.01 in valore assoluto. La soglia deve essere fissata
prima delle run finali e non adattata osservando il test.

## 11. Protocollo di benchmark

### 11.1 Condizioni

- una sola T4;
- batch 1;
- input fisso 1 x 3 x 512 x 512;
- eval/inference mode;
- nessun processo concorrente controllabile;
- warmup prima delle misure;
- sincronizzazione CUDA corretta;
- campioni reali normalizzati, non solo input zero;
- stessa sessione e stesso ordine delle precisioni quando possibile;
- almeno tre trial.

### 11.2 Due misure separate

Model-only:

- input già residente su GPU;
- misura GPU Compute Time;
- metrica primaria per confrontare le precisioni.

Host path:

- H2D;
- enqueue/inference;
- D2H dei logits o dell'output dichiarato;
- niente caricamento Hugging Face, decodifica PIL o plotting.

Il tempo di DataLoader e preprocessing può essere riportato separatamente, ma
non deve essere sommato implicitamente alla latenza del modello.

### 11.3 Statistiche

Salvare per ogni trial e aggregato:

- numero di warmup e misure;
- min e max;
- mean e standard deviation;
- median/p50;
- p90, p95 e p99;
- FPS derivati dalla mediana e dalla media;
- speedup rispetto a TensorRT FP32;
- throughput batch 1;
- dimensione engine;
- memoria GPU, quando misurabile in modo affidabile;
- temperatura, clock e potenza come metadati diagnostici, non come risultato
  assoluto se Kaggle non permette un controllo stabile.

Usare trtexec come misura di controllo e un runner Python TensorRT per
integrare valutazione e input reali. I due strumenti devono dichiarare se
includono o escludono i trasferimenti.

## 12. Artefatti

### 12.1 Training FP32 standard

Non cambiare i file esistenti:

    history.json
    test_metrics.json
    best.ckpt
    last.ckpt
    prediction_grid.png

Aggiungere:

    run_manifest.json
    environment.json
    artifacts_manifest.json
    upload_status.json

### 12.2 PTQ

    run_manifest.json
    source_checkpoint.json
    calibration_manifest.json
    environment.json
    model_fp32.onnx
    model_int8_qdq.onnx
    engine_fp32.plan
    engine_fp16.plan
    engine_int8.plan
    test_metrics_fp32.json
    test_metrics_fp16.json
    test_metrics_int8.json
    export_validation.json
    quantization_summary.json
    benchmark_results.json
    comparison_metrics.json
    artifacts_manifest.json
    upload_status.json
    logs/

### 12.3 QAT

    history.json
    test_metrics.json
    best.ckpt
    last.ckpt
    qat_state/
    calibration_manifest.json
    model_int8_qat_qdq.onnx
    engine_int8_qat.plan
    export_validation.json
    quantization_summary.json
    benchmark_results.json
    run_manifest.json
    environment.json
    artifacts_manifest.json
    upload_status.json

### 12.4 Manifest degli artefatti

Per ogni file:

- path relativo;
- tipo logico;
- dimensione in byte;
- SHA-256;
- timestamp;
- obbligatorio oppure opzionale;
- stato upload;
- Google Drive file ID, se caricato.

Non includere token, client secret o refresh token.

## 13. Google Drive

### 13.1 Modello di autorizzazione

- Account principale: proprietario della cartella e identità OAuth usata da
  Kaggle.
- Secondo account: accesso tramite normale condivisione Drive.
- Viewer è sufficiente per scaricare e analizzare.
- Editor è necessario solo se il secondo account deve scrivere nella stessa
  cartella.
- Nessun service account.
- Nessuno Shared Drive richiesto.

### 13.2 Setup una tantum

Documentare:

1. creazione di un progetto Google Cloud;
2. abilitazione Google Drive API;
3. creazione client OAuth desktop;
4. consenso dell'account proprietario con accesso offline;
5. creazione della cartella app-managed;
6. recupero del folder ID;
7. inserimento in Kaggle Secrets di:
   - GDRIVE_CLIENT_ID;
   - GDRIVE_CLIENT_SECRET;
   - GDRIVE_REFRESH_TOKEN;
   - GDRIVE_FOLDER_ID;
8. condivisione manuale della cartella col secondo account.

Preferire lo scope drive.file. Se non consente di usare una cartella
preesistente, far creare la cartella radice all'integrazione durante il setup e
condividerla successivamente.

### 13.3 Upload

Il notebook deve:

1. scrivere tutto in /kaggle/working;
2. chiudere i file;
3. calcolare hash;
4. creare la sottocartella remota con RUN_ID;
5. usare upload resumable per checkpoint, ONNX ed engine;
6. applicare retry con backoff;
7. non sovrascrivere una run remota esistente;
8. registrare Drive file ID e checksum;
9. creare upload_status.json;
10. mantenere sempre la copia locale.

Un fallimento Drive non deve cancellare né invalidare i risultati locali.
Deve produrre un warning evidente e uno stato failed o partial.

### 13.4 Download

PTQ e QAT devono poter recuperare una source run da:

- percorso locale/Kaggle Input;
- Drive folder ID.

Il download deve:

- leggere prima il manifest;
- scaricare solo i file richiesti;
- supportare download resumable per best.ckpt;
- verificare SHA-256;
- fallire se source_run_id, modello o hash non corrispondono.

### 13.5 Sicurezza

- Leggere segreti con Kaggle UserSecretsClient.
- Non stampare valori o oggetti credentials.
- Non salvare segreti in notebook output, log, JSON o checkpoint.
- Redigere errori HTTP che possano includere header Authorization.
- DRIVE_UPLOAD_ENABLED deve avere default False.
- Il notebook deve restare eseguibile senza credenziali Drive.

## 14. Struttura remota

    S5Mars_Experiments/
      training/
        <model_run_name>/
          <run_id>/
      ptq/
        <model_run_name>/
          <source_run_id>/
            <ptq_run_id>/
      qat/
        <model_run_name>/
          <source_run_id>/
            <qat_run_id>/

Il secondo account può aggiungere una scorciatoia alla cartella condivisa nel
proprio My Drive.

## 15. Componenti riusabili

Anche se i notebook Kaggle restano autosufficienti, mantenere interfacce
coerenti per funzioni duplicate:

- collect_environment();
- build_run_manifest();
- sha256_file();
- build_artifacts_manifest();
- get_drive_credentials();
- upload_run_directory();
- download_source_run();
- select_calibration_indices();
- export_onnx();
- validate_export();
- build_tensorrt_engine();
- evaluate_runtime();
- benchmark_runtime();

La duplicazione tra notebook deve essere minimizzata durante la manutenzione:
quando una funzione condivisa cambia, aggiornare entrambe le copie e testarne
gli output. Non introdurre un'importazione da src che renda il notebook non
autosufficiente su Kaggle.

## 16. Test

### 16.1 Unit test locali

Aggiungere test per:

- validazione configurazione PTQ;
- QAT flag e campi obbligatori;
- selezione deterministica degli indici di calibrazione;
- subset annidati 32/128/512;
- serializzazione manifest;
- hash degli artefatti;
- statistiche benchmark su timings noti;
- calcolo speedup;
- Drive uploader con client mock;
- retry e partial upload;
- redazione dei segreti;
- ricostruzione model config per SegFormer, SMP e LCNet.

Non richiedere TensorRT o GPU nei test unitari locali.

### 16.2 Notebook validation

- JSON valido per entrambi i notebook;
- nessun execution output contenente token;
- nessun placeholder interpretato come credenziale valida;
- celle in ordine eseguibile top-to-bottom;
- configurazione centrale unica;
- codice condizionale QAT non eseguito in modalità none.

### 16.3 Smoke test Kaggle T4

Modalità rapida:

    CALIBRATION_SIZE = 4
    LIMIT_VAL_BATCHES = 2
    LIMIT_TEST_BATCHES = 2
    BENCHMARK_WARMUP_ITERATIONS = 5
    BENCHMARK_MEASUREMENT_ITERATIONS = 10
    DRIVE_UPLOAD_ENABLED = False

Verificare almeno:

- export ONNX;
- build TensorRT FP32 e FP16;
- una build INT8;
- output shape;
- file metriche;
- benchmark JSON;
- manifest e hash.

Ripetere lo smoke per almeno un modello per famiglia architetturale prima delle
run complete.

### 16.4 Test Drive

Usare una run piccola e una cartella di test:

- upload JSON piccolo;
- upload resumable di un file maggiore di 5 MB;
- download e verifica hash;
- accesso dal secondo account;
- simulazione errore rete;
- verifica che la copia locale rimanga;
- verifica che nessun segreto appaia negli output.

## 17. Criteri di accettazione

### PTQ

- Il notebook esegue top-to-bottom con un checkpoint valido.
- Ricostruisce automaticamente tutte le famiglie di modello.
- Rifiuta checkpoint o manifest incompatibili.
- FP16 non accede al calibration loader.
- INT8 usa soltanto train per calibrazione.
- TensorRT FP32 viene prodotto prima dei confronti ridotti.
- Le metriche esistenti sono riutilizzate senza modificarne lo schema.
- Ogni engine ha benchmark, precision coverage e manifest.
- Il benchmark usa una sola T4 e batch 1.
- Gli output locali restano disponibili se Drive fallisce.

### QAT

- QUANTIZATION_MODE=none mantiene il percorso attuale.
- QUANTIZATION_MODE=qat_int8 richiede un best FP32 checkpoint.
- QAT non parte da pesi casuali nel core.
- Il best checkpoint QAT è selezionato tramite validation.
- Lo stato di quantizzazione è ripristinabile.
- Viene prodotto un vero engine TensorRT INT8.
- Le metriche fake-quant e TensorRT non vengono confuse.

### Drive

- L'account proprietario carica nella cartella configurata.
- Il secondo account vede e scarica gli artefatti.
- Upload e download verificano hash.
- Nessun segreto è persistito.
- Run con stesso ID non viene sovrascritta silenziosamente.

## 18. Ordine di implementazione

### Milestone 0 - Contratto degli artefatti

- Definire schema versionato di run_manifest e artifacts_manifest.
- Aggiungere RUN_ID e directory univoche.
- Salvare configurazione completa e hash best.ckpt.
- Mantenere history.json e test_metrics.json invariati.

Gate: una run FP32 esistente produce un bundle ricostruibile.

### Milestone 1 - Google Drive

- Implementare OAuth setup documentato.
- Implementare secrets, upload e download.
- Aggiungere hash, retry e upload_status.
- Verificare accesso dal secondo account.

Gate: una run smoke viene caricata e recuperata senza perdita.

### Milestone 2 - Export e TensorRT FP32

- Creare scheletro del notebook PTQ.
- Caricare source run.
- Export e validazione ONNX per tutte le famiglie.
- Build/evaluate/benchmark TensorRT FP32.

Gate: TensorRT FP32 è numericamente e metricamente coerente con PyTorch.

### Milestone 3 - FP16

- Build TensorRT FP16.
- Report precision coverage.
- Valutazione e benchmark.

Gate: risultati FP16 confrontabili con TensorRT FP32.

### Milestone 4 - PTQ INT8 core

- Calibration sampler random deterministico, 128 immagini.
- ModelOpt PTQ e Q/DQ export.
- Build, coverage, validation, test e benchmark.

Gate: una run INT8 completa produce tutti gli artefatti obbligatori.

### Milestone 5 - Ablation PTQ

- Calibration size 32 e 512.
- Class-coverage 128.
- Seed calibration multipli.
- Recovery selettivo solo dove necessario.

Gate: tabella comparativa generabile dai soli JSON salvati.

### Milestone 6 - QAT

- Flag e precondizioni.
- Caricamento source FP32.
- Inizializzazione fake quant e fine-tuning.
- Salvataggio stato QAT.
- Export Q/DQ, TensorRT INT8 e benchmark.

Gate: confronto PTQ-vs-QAT sullo stesso checkpoint sorgente.

### Milestone 7 - Consolidamento

- Regression test modalità none.
- Smoke test di tutte le famiglie.
- Documentazione README.
- Pulizia dei notebook output.
- Congelamento versioni Kaggle funzionanti.

Gate: piano sperimentale eseguibile senza modifiche manuali fuori dalla cella
di configurazione.

## 19. Strategia di esecuzione consigliata

1. Addestrare e conservare i checkpoint FP32 finali.
2. Eseguire TensorRT FP32, FP16 e PTQ INT8 core per tutti.
3. Costruire il primo fronte di Pareto.
4. Eseguire ablation di calibrazione solo sui modelli rappresentativi o
   problematici.
5. Applicare QAT almeno ai modelli in cui PTQ perde accuratezza e a un modello
   PTQ-robusto usato come controllo.
6. Analizzare i JSON su Drive senza scegliere nuove configurazioni dal test.
7. Eseguire le valutazioni test finali delle configurazioni congelate.

## 20. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Export non supportato per alcuni operatori | gate ONNX/TensorRT FP32 prima della quantizzazione |
| FP16/INT8 solo parzialmente applicati | precision coverage obbligatoria |
| SegFormer sensibile a INT8 | mixed precision, SmoothQuant o QAT |
| LCNet con SELU/sigmoid poco fondibile | report per-layer e fallback esplicito |
| Overhead Q/DQ annulla lo speedup | benchmark vero engine, non fake quant |
| Versioni Kaggle cambiano | environment manifest e smoke test |
| Test leakage | selection solo su validation |
| Run sovrascritte | RUN_ID e no-overwrite |
| Upload Drive interrotto | resumable upload, retry e copia locale |
| Token esposti | Kaggle Secrets, redazione e test |
| Ranking T4 non trasferibile al rover | conclusioni limitate alla piattaforma proxy |

## 21. Riferimenti tecnici primari

- PyTorch/torchao PT2E quantization:
  https://docs.pytorch.org/ao/stable/pt2e_quantization/
- NVIDIA Model Optimizer, PTQ e QAT:
  https://nvidia.github.io/Model-Optimizer/
- TensorRT, quantizzazione esplicita Q/DQ:
  https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
- TensorRT performance benchmarking:
  https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html
- ONNX Runtime FP16:
  https://onnxruntime.ai/docs/performance/model-optimizations/float16.html
- Google OAuth offline access:
  https://developers.google.com/identity/protocols/oauth2/web-server
- Google Drive resumable uploads:
  https://developers.google.com/workspace/drive/api/guides/manage-uploads

# 22. Note dell'utente

Te non devi avviare nessun training, lo farò poi io utente manualmente su kaggle. Te devi solo implementare il notebook PTQ e le modifiche al notebook di training per QAT, senza cambiare la logica di training FP32 esistente.
