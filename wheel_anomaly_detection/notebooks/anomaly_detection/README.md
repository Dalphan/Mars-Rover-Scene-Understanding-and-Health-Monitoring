# Notebook anomaly detection

Questa directory contiene il notebook end-to-end per Kaggle.

`kaggle_wheel_anomaly_detection.ipynb` è autosufficiente: contiene una copia
esplicita del codice per Dataset, preprocessing, audit e DataLoader e non importa
`src/anomaly_detection/`. Su Kaggle è quindi sufficiente collegare il dataset
già estratto.

La duplicazione è intenzionale, ma introduce il rischio che notebook e moduli
divergano. Ogni modifica al caricamento dei dati deve essere riportata in
entrambi. `PATCHCORE_PRESET="light"` seleziona ResNet-18, frame `384x512` ed
embedding 384; `"reference"` seleziona WideResNet-50-2, resize lato corto 256,
center crop 224 ed embedding `1024 -> 1024`. Il notebook implementa
fit, coreset, calibrazione, prediction, validation, test e salvataggio degli
artefatti senza importare i moduli del repository. Le augmentation train sono
disabilitate per PatchCore. Tutti i parametri modificabili, incluse metriche e
upload Google Drive opzionale, sono raccolti nella cella `Configuration`.

Dopo setup, import e configurazione, il notebook dichiara tutte le classi e le
funzioni prima di avviare la pipeline. Le celle di esecuzione finali seguono
l'ordine: inizializzazione, preparazione e verifica dati, audit opzionale,
costruzione modello, fit, validation, test e persistenza degli artefatti. Fit e
valutazioni sono separati per rendere più semplice riprendere un run Kaggle.
Tra test e persistenza viene eseguita la visualizzazione qualitativa di esempi
clean e anomali; genera pannelli input, ground truth, anomaly map e overlay e
salva le figure PNG insieme agli altri artefatti del run.


La cella `diagnostics-run` mostra e salva anche distribuzioni degli score raw,
casi clean/hole estremi, Image AUROC/Image AP e pixel AP per sottogruppo,
baseline della pixel AP e AUPRO/PRO ai cutoff FPR `0.05`, `0.10` e `0.30`. Le
diagnostiche sono raccolte durante la normale valutazione, senza un secondo
passaggio completo del modello.

`kaggle_wheel_anomaly_detection_model_roadmap.ipynb` � una copia separata che
aggiunge EfficientAD-S, SuperSimpleNet e TinyGLASS; l'originale resta invariato.
Nella copia si cambia esecuzione con `MODEL_NAME`; le configurazioni full
restano raccolte in `MODEL_CONFIGS`. Anche la copia è autosufficiente e non
importa `src`. EfficientAD scarica il teacher fissato di `nelson1425`, richiede
un secret Kaggle `HF_TOKEN` abilitato a `ILSVRC/imagenet-1k` e supporta resume
tramite `RESUME`/`RESUME_RUN_DIR`. TinyGLASS scarica DTD R1.0.1 dalla fonte VGG nella directory di lavoro, lo
estrae in modo sicuro e usa `dtd/images/` dopo aver verificato le 5.640 immagini.

I tre modelli condividono output, runner, metriche e checkpoint. Il train resta
clean-only. EfficientAD esegue sempre 70.000 step e usa soltanto i campioni clean
di validation per i quantili finali, senza model selection sulle anomalie.
SuperSimpleNet e TinyGLASS usano weights torchvision espliciti
`IMAGENET1K_V1`. `FIXED_TRAINING_DURATION=True` disabilita la model selection
intermedia e mantiene il numero completo di epoche; impostandolo a `False` si
riabilita la selezione del best checkpoint sulla validation. Train batch size ed
evaluation batch size sono proprietà del modello.
