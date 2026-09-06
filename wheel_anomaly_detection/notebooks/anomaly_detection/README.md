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
un secret Kaggle `HF_TOKEN` abilitato a `ILSVRC/imagenet-1k`. I tre modelli
trainabili supportano resume tramite `RESUME`/`RESUME_RUN_DIR`: EfficientAD a
intervalli di step, SuperSimpleNet e TinyGLASS a fine epoca. TinyGLASS scarica
DTD R1.0.1 dalla fonte VGG soltanto con `las_mode="texture"` o `"mixed"`;
la modalita `"hole"` non usa texture esterne.

I tre modelli condividono output, runner, metriche e checkpoint. Il train resta
clean-only. Il preset EfficientAD corrente usa tutte le clean di training con
input `384x384` per 70.000 step fissi, `global_max` senza ROI e la clean validation soltanto per i
quantili globali finali q90/q99.5. La variante pose-conditioned spaziale resta
disponibile ma non e il default del notebook.
Il fit EfficientAD usa mixed precision FP16 con `GradScaler` su CUDA, unisce
originale/augmentata nel forward del teacher e
originale/augmentata/ImageNet nel forward dello student. Hard mining e loss
restano FP32. Lo stato dello scaler entra nel checkpoint, salvato ogni 5.000
step; su CPU AMP viene disabilitato automaticamente.
Con `EFFICIENTAD_CALIBRATION_ABLATION_ENABLED=True`, nella variante spaziale una cella post-fit valuta
lo stesso checkpoint nelle modalita `global`, `global_roi` e `spatial_roi`,
sempre con score `max`. Salva `efficientad_calibration_ablation.json` con le
metriche complete e AUPRO per validation e test, senza selezionare una modalita
e ripristinando al termine la configurazione scelta durante il fit.
Con `EFFICIENTAD_GLOBAL_TOPK_ABLATION_ENABLED=True`, il preset globale
confronta sullo stesso checkpoint il massimo ridimensionato, il massimo nativo
e le medie top-k dei migliori `2, 4, 8, 16` pixel della mappa nativa. La scelta
usa soltanto l'Image AUROC di validation e sul test viene valutato solo il
vincitore. Il file `efficientad_global_topk_ablation.json` include anche il
controllo d'invarianza delle metriche di mappa.
SuperSimpleNet e TinyGLASS usano weights torchvision espliciti
`IMAGENET1K_V1`. `FIXED_TRAINING_DURATION=True` mantiene SuperSimpleNet a durata
fissa; TinyGLASS usa separatamente `TINYGLASS_FIXED_TRAINING_DURATION=False`,
seleziona il best checkpoint sulla validation, riusa a ogni controllo una cache
CPU delle feature del backbone congelato e applica early stopping configurabile
in `MODEL_CONFIGS`. La cache non viene salvata nel checkpoint e viene ricostruita
una sola volta dopo un resume; storico, patience e best state sono invece
ripristinati. Train batch size ed evaluation batch size sono proprietà del
modello. Dopo la valutazione standard, la roadmap confronta sullo stesso
checkpoint le aggregazioni definite in `TINYGLASS_IMAGE_SCORE_CANDIDATES`.
Seleziona la migliore soltanto sulla validation, valuta sul test solo quella
selezionata e salva `tinyglass_image_score_aggregation.json`; non modifica il
training, le anomaly map o le metriche pixel.
La cella successiva esegue anche l'ablazione diagnostica
TINYGLASS_GAUSSIAN_SIGMA_CANDIDATES=(0, 1, 2, 4) esclusivamente sulla
validation. Non sceglie il sigma e non valuta i candidati sul test; salva
metriche pixel, target-wheel e AUPRO in
tinyglass_gaussian_sigma_ablation.json, controlla l'invarianza delle metriche
image-level e ripristina sempre il sigma della configurazione.

TinyGLASS mantiene `freeze_backbone=True` come default. Impostando
`TINYGLASS_FINE_TUNE_EXPERIMENT=True` la cella Configuration avvia invece un
nuovo run da 20 epoche con backbone LR `1e-5`, patience 7 e cache validation
disabilitata. Il backbone passa in train mode, entra in un gruppo optimizer
separato ed è incluso in training checkpoint e best state. Non usare `RESUME`
per passare da frozen a fine-tuned: la configurazione checkpoint rifiuta il
cambio.

La configurazione corrente della copia roadmap seleziona TinyGLASS sulla sola
posa `A_overhead`, con crop 720x720 e input 384x384. Mantiene layer2, che
produce una griglia 48x48, sigma 1, LAS hole `70/25/5`, backbone frozen, LR
`5e-5` e patience 40. Il cambio di input è incluso nel contratto checkpoint:
avviare un nuovo run con `RESUME=False`, senza riprendere il checkpoint
256x256.
