# Notebook anomaly detection

Questa directory contiene il notebook end-to-end per Kaggle.

## Figura 6.1: confronto qualitativo dei sei modelli

`kaggle_wheel_anomaly_qualitative_comparison.ipynb` prepara griglie 2x4 con RGB,
ground truth, PC-S, PC-L, EAD-S, SSN, TG-384 e TG-512 sugli stessi campioni di
test. Riprende il flusso del notebook qualitativo S5Mars: 20 immagini casuali
senza ripetizioni, seed 42, un modello alla volta, navigatore e scelta della
figura per la tesi. Esporta `figure_6_1.png`, `figure_6_1.pdf`, pannelli,
mappe numeriche e manifest; lo ZIP comprende tutti i confronti ed esclude i
checkpoint. Non modifica i capitoli LaTeX.

Collegare il dataset sintetico estratto, attivare GPU e Internet e configurare
i secret Drive già usati dagli altri notebook. `LOCAL_CHECKPOINTS` permette di
usare checkpoint finali locali. I quattro modelli appresi vengono ripristinati
con i controlli dei notebook di efficienza, senza training o ricalibrazione;
TG-512 conserva il restore stretto e l'override sigma 4. Le definizioni dei
modelli e del preprocessing sono copie esplicite dei sorgenti autorevoli.

Se manca un checkpoint PatchCore, il notebook costruisce la memory bank sulle
2.800 immagini normali di training posa A, con reservoir 1M, cap 512 patch per
immagine e bank 50k per entrambi i preset. Salva il checkpoint subito dopo il
fit e lo riusa nelle esecuzioni successive. Per evitare un nuovo fit dopo la
chiusura della sessione, scaricare separatamente i due checkpoint e ricollegarli
tramite `LOCAL_CHECKPOINTS`. Un checkpoint fornito ma incompatibile genera un
errore, senza avviare automaticamente un fit sostitutivo.

Le mappe sono continue, senza soglia binaria, e vengono mostrate sulla scala
normalizzata originale `[0,1]`, come negli altri notebook anomaly. La precedente
riscalatura del contrasto tramite quantili di validation è stata rimossa.
Il rendering legge direttamente le mappe numeriche con interpolazione bilineare;
la ground truth binaria conserva l'interpolazione nearest. Mantiene lo smoothing
già applicato dal modello: sigma 4 per PatchCore e SSN, sigma 1 per TG-384 e
sigma 4 per TG-512; EfficientAD mantiene la propria interpolazione finale.
Le mappe native TinyGLASS, precedenti allo smoothing, restano salvate separatamente.
I valori normalizzati non sono probabilità calibrate confrontabili tra modelli.
Il bordo non osservato da PC-L è grigio nel crop comune 720x720; la sua regione
osservata è 630x630. Gli output della versione precedente conservano campioni
e checkpoint PatchCore: rieseguire il notebook aggiorna le mappe e le figure
riusando le memory bank già salvate.

Verifica locale: `python -m unittest tests.test_ad_qualitative_notebook -v`.
I controlli coprono sincronizzazione dei sorgenti, profili e restore,
allineamento spaziale, fit PatchCore minimo, ripresa delle predizioni e
rendering PNG/PDF; il fit completo e i checkpoint Drive si eseguono su Kaggle.

## EfficientAD-S: solo efficienza del checkpoint selezionato

`kaggle_efficientad_efficiency.ipynb` carica il checkpoint finale del run
`efficientad_s_A_overhead_crop720_global_fixed70k_balanced__seed42__20260901_120051`:
input 256x256, posa A, crop `(90, 240, 720, 720)`, 384 canali, calibrazione
globale e score `global_max`. Ripristina teacher, student, autoencoder e
statistiche di calibrazione senza training, ricalibrazione, ablation o nuova
valutazione di accuracy. Gli input restano RGB in `[0,1]`: la normalizzazione
ImageNet è interna al modello e non viene applicata dal loader.

Il checkpoint verificato precede i campi `input_size` e `mixed_precision` del
contratto attuale. Il restore consente e registra solo questi due default di
compatibilità: input 256 ricavato dal manifest e `mixed_precision=True` come
default del costruttore relativo al training, non come precisione di inferenza
né evidenza della precisione originale del training. Configurazione, tensori e
stato global-max vengono altrimenti verificati strettamente.

Su Kaggle usare una sessione nuova con Tesla T4 e il dataset sintetico estratto.
Il download Drive usa i secret `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`,
`GDRIVE_REFRESH_TOKEN`; è disponibile anche `CHECKPOINT_SOURCE="local"`.
Il teacher completo è già nel checkpoint: non servono `HF_TOKEN`, ImageNet o
un download di pesi separato. Il benchmark riusa l'helper autorevole
`src/anomaly_detection/evaluation/efficiency.py` con FP32, batch 1, zero worker,
10 warm-up e 50 misurazioni per ambito. Salva JSON e CSV sotto
`/kaggle/working/efficientad_efficiency/`, senza upload o overwrite del
checkpoint. Come negli altri benchmark, l'end-to-end include anche I/O e
trasformazioni delle maschere e la memoria è quella dell'allocatore CUDA;
la misura energetica è invece una fase predict-only separata, con input già
residente in GPU: 5 s di warm-up e 3 finestre da 30 s. Preferisce il contatore
NVML in mJ; in assenza di un contatore utilizzabile integra le letture di
potenza NVML ogni 0,1 s, senza mescolare backend tra le ripetizioni. Salva
J/image, potenza media, dispersione tra le ripetizioni e dettaglio dei sensori
in `inference_energy_batch1.json`, oltre a JSON/CSV del benchmark. Il valore
è l'energia dell'intera GPU, non della CPU, del processo o del rover, e il
consumo a riposo non viene sottratto. Dispositivi CUDA/NVML sono allineati per
UUID; sensori non accessibili o misure parziali vengono segnalati con un motivo
esplicito e aggregato nullo. Non vengono usati TDP o latenza come stime
energetiche. `ENERGY_ENABLED`, `ENERGY_METHOD` e le durate sono configurabili;
`nvidia-ml-py` viene installato solo se necessario. La latenza/memoria restano
nel protocollo precedente. TinyGLASS, PatchCore e SuperSimpleNet usano ora lo
stesso helper e le stesse finestre energetiche. Le copie dei notebook vengono verificate durante la preparazione;
i file temporanei di test creati per questi notebook vengono poi rimossi.

## TinyGLASS: efficienza TG-384 e test TG-512 opzionale

`kaggle_tinyglass_efficiency.ipynb` misura per default l'efficienza del TG-384
già addestrato: 384x384, layer2, sigma 1, ResNet-18 congelata, posa A e crop
fisso `(90, 240, 720, 720)`. Il checkpoint Drive è quello del run seed42 del
31 agosto 2026, `20260831_112751`; la configurazione effettiva è stata verificata
nel suo `run_manifest.json`. Il restore usa i parametri salvati e verifica
input, layer, sigma, LAS hole 70/25/5 e preprocessing. Non esegue fit o
calibrazione. Accetta anche un checkpoint finale locale con
`CHECKPOINT_SOURCE="local"` e `MODEL_CHECKPOINT_PATH`.

I default sono `REFERENCE_PROFILE="TG-384"`, `BENCHMARK_ENABLED=True`,
`TEST_EVALUATION_ENABLED=False` ed `ENERGY_ENABLED=True`. Input, layer, sigma e
ID Drive sono associati al profilo: non basta ridimensionare il TG-512 per
misurare il modello TG-384 addestrato. Il profilo `"TG-512"` conserva il
checkpoint 512/layer3 e applica sigma 4 dopo il restore. Per ripeterne il test,
selezionare quel profilo, impostare `TEST_EVALUATION_ENABLED=True` e
`BENCHMARK_ENABLED=False`. Eseguire un solo profilo per sessione nuova.

Su Kaggle collegare il dataset sintetico estratto e selezionare Tesla T4.
Per Drive servono Internet e i secret `GDRIVE_CLIENT_ID`,
`GDRIVE_CLIENT_SECRET`, `GDRIVE_REFRESH_TOKEN`; la modalità locale non richiede
Drive. Non occorre scaricare pesi ImageNet o texture DTD.

L'efficienza usa la sola validation e lo stesso helper autorevole
`src/anomaly_detection/evaluation/efficiency.py`: FP32, batch 1, 10 warm-up e
50 misurazioni per ambito, loader sequenziale con zero worker. Distingue
predict-only, trasferimento+predict ed end-to-end; quest'ultimo include I/O e
trasformazione delle annotazioni. La Gaussian smoothing del profilo è inclusa
in predict. Salva latenza, throughput, parametri, checkpoint, memoria CUDA
totale e incrementale, configurazioni e provenienza. Discrepanze del manifest
o dei conteggi validation sono registrate e segnalate, senza filtrare immagini.

L'energia GPU usa 5 s di warm-up e 3 finestre da 30 s, contatore NVML oppure
integrazione delle letture di potenza, con lo stesso scope di EfficientAD.
Salva J/image, potenza media, variabilità e backend. Sensori indisponibili o
errori parziali producono un valore nullo con motivo. Leggere il JSON finale
salvato dopo questa fase, che richiede circa 95 s. I tre file da scaricare sono
`inference_benchmark_batch1.json`, `inference_benchmark_batch1.csv` e
`inference_energy_batch1.json`, nella nuova cartella sotto
`/kaggle/working/tinyglass_efficiency/`; il nome e i report identificano
384x384, layer2 e sigma 1. Nessun upload o risalvataggio del checkpoint.

Il test TG-512 opzionale conserva le 808 immagini posa A (404 coppie
clean/hole), senza augmentation o filtri sui campioni. Verifica checksum del
manifest, conteggi, pairing e metadati; ricontrolla checkpoint e manifest dopo
l'inferenza. Riusa gli accumulatori autorevoli per image/pixel AUROC/AP,
sottogruppi e AUPRO ai cutoff FPR 0,05/0,10/0,30. Le clean ereditano la severità
della hole abbinata; 2.048 bin pixel e 256 PRO limitano la memoria. I file test
restano in `/kaggle/working/tinyglass_efficiency/test_sigma4/`, con score
frazionari in JSON/CSV e percentuali nella stampa.

## SuperSimpleNet: solo efficienza del checkpoint selezionato

`kaggle_supersimplenet_efficiency.ipynb` ripristina il checkpoint finale
selezionato nel catalogo real-wheel: WideResNet-50-2, input 256x256, posa A,
crop `(90, 240, 720, 720)`, Perlin 0.2, anomalie sintetiche non ristrette alla
ruota, score top-1% sulla mappa nativa e smoothing sigma 4. Non sostituisce il
preset corrente con restrizione alla ruota: verifica i campi del run scelto,
la configurazione salvata e tutti i tensori con restore stretto. Non esegue
training, calibrazione, accuracy, selezione dello score o modifica del
checkpoint. Il forward resta completo, incluso il ramo decisionale ancora
eseguito dall'implementazione autorevole anche con lo score top-1%.

Usare una sessione Kaggle nuova con Tesla T4, il dataset sintetico estratto e
i secret Drive `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`,
`GDRIVE_REFRESH_TOKEN`. Il checkpoint è identificato da folder/file ID nel
notebook; in alternativa sono disponibili `CHECKPOINT_SOURCE="local"` e
`MODEL_CHECKPOINT_PATH`. Tutti i pesi sono già nel checkpoint: non serve
scaricare ImageNet. Preprocessing RGB, crop fisso, resize 256 e normalizzazione
ImageNet coincidono con il run selezionato; solo validation posa A, senza
augmentation, filtering o audit completo delle maschere.

Il protocollo coincide con gli altri notebook dedicati: FP32, batch 1,
zero worker, 10 warm-up e 50 misurazioni per ambito per latenza/memoria;
energia predict-only separata, input residente in GPU, 5 s di warm-up e
3 finestre da 30 s. Usa lo stesso helper autorevole
`src/anomaly_detection/evaluation/efficiency.py`: contatore NVML oppure
integrazione delle letture di potenza a 0,1 s, senza stime TDP o da latenza.
L'energia è device-wide, senza sottrazione dell'idle e senza CPU/loader/transfer;
sensori indisponibili o misure parziali producono un valore nullo con motivo.

Salva `inference_benchmark_batch1.json`, `inference_benchmark_batch1.csv` e
`inference_energy_batch1.json` sotto
`/kaggle/working/supersimplenet_efficiency/`, con J/image, potenza media,
dispersione tra ripetizioni, backend e dettaglio dei sensori. Leggere il JSON
finale dopo la fase energetica; il print precedente di latenza può avere
ancora energia nulla. Nessun upload o risalvataggio del modello.

## PatchCore Reference sulla sola posa A

`kaggle_patchcore_reference_pose_a.ipynb` è un run dedicato e autosufficiente;
non modifica gli altri notebook. Filtra `A_overhead` in train, validation e test,
applica il crop fisso `(90, 240, 720, 720)`, poi conserva la geometria Reference
resize lato corto 256 e center crop 224, con normalizzazione ImageNet e senza
augmentation. I conteggi attesi sono 2.800 clean train, 296 clean + 96 hole in
validation e 404 clean + 404 hole in test.

Il backbone è WideResNet-50-2, con embedding 1024, un vicino, sigma 4 e bank
50.000. Il reservoir è limitato a 1.000.000 di candidati; il cap è aumentato a
512 patch per immagine perché 128 patch sulle 2.800 clean, con ratio 0,1,
produrrebbero solo 35.840 elementi. Il notebook verifica la dimensione reale
della bank dopo il fit. Questa modifica al campionamento viene registrata e
non rende il nuovo run una replica esatta del precedente run multi-posa.
La selezione greedy di 50.000 elementi è costosa: serve una sessione GPU full
con RAM host sufficiente, non un fit su CPU o una bank ridotta di smoke test.

Prima del fit usa lo stesso audit visivo non bloccante degli altri notebook.
La scansione numerica aggiuntiva delle maschere è disabilitata per default
(`MASK_VISIBILITY_AUDIT_ENABLED=False`). Se abilitata, salva
`preprocessing_validity_audit.json` e segnala anche le maschere vuote, senza
interrompere il fit, eliminare campioni o cambiare le label. Il center crop
Reference può escludere fori ai bordi: questo resta un limite del preprocessing
che può influire sulle metriche, non viene corretto cambiando la geometria
o selezionando campioni in base alle annotazioni di validation/test.
Dopo il fit salva subito `model.ckpt` e `fit_summary.json`,
poi esegue il benchmark batch 1 con 10 warm-up e 50 misurazioni per ambito.
`inference_benchmark_batch1.json` distingue predict-only su input già residente
dalla pipeline sequenziale (che include anche lettura/trasformazione delle
maschere), e riporta latenza, throughput, memoria CUDA totale e incrementale,
dimensione della bank/checkpoint e metadati hardware/software. La fase
energetica separata usa lo stesso protocollo di EfficientAD/TinyGLASS:
5 s di warm-up, 3 finestre da 30 s, FP32, batch 1, input gia residente in GPU,
contatore NVML o integrazione della potenza misurata. Salva subito
`inference_energy_batch1.json` e aggiorna il JSON di latenza con il riepilogo
energetico, anche senza rieseguire successivamente accuracy e diagnostiche.
La misura e dell'intera GPU, senza CPU/loader/trasferimento e senza sottrarre
l'idle. `ENERGY_ENABLED` e indipendente da `BENCHMARK_ENABLED`.

Se la sessione con il modello gia fit e ancora attiva, eseguire soltanto le
celle aggiornate `batch-one-efficiency-definitions`, `gpu-energy-dependencies`
e `gpu-energy-run`: i default energetici sono disponibili anche senza
rieseguire Configuration e cambiare RUN_ID/OUTPUT_DIR. Non ripetere fit,
inizializzazione o configurazione solo per aggiungere energia. In una sessione
nuova l'esecuzione di tutte le celle mantiene il fit clean-only originale.
Per confronti usare lo stesso dispositivo e lo stesso ambito temporale.
L'helper autorevole vive in `src/anomaly_detection/evaluation/efficiency.py`;
il notebook ne contiene una copia esplicita verificata durante i controlli.

Gli artefatti finali includono metriche primarie e audit, diagnostiche AUPRO/PRO,
score e figure. Sono salvati in una cartella unica sotto `/kaggle/working/`;
l'upload Drive è opzionale e disabilitato per default. Per il run collegare il
dataset estratto, attivare GPU e Internet per i pesi ImageNet. TinyGLASS sigma 4
non viene rieseguito: restano validi i risultati locali già acquisiti.

## Notebook generali e pilot reale

`kaggle_wheel_anomaly_detection.ipynb` è autosufficiente: contiene una copia
esplicita del codice per Dataset, preprocessing, audit e DataLoader e non importa
`src/anomaly_detection/`. Su Kaggle è quindi sufficiente collegare il dataset
già estratto.

`kaggle_real_wheel_zero_shot.ipynb` prepara invece il pilot MAHLI held-out:
usa direttamente `/kaggle/input/datasets/dalphan01/real-mars-rover-wheel-ad/real_wheel_pilot_v2`, unisce manifest,
crop e maschere approssimative,
applica crop per immagine e resize allineato e mostra i due gate visivi.
`MODEL_NAME` seleziona PatchCore, EfficientAD-S, SuperSimpleNet o TinyGLASS e
deriva input e normalizzazione dal preset corrispondente. Le implementazioni
sono copiate dalla roadmap; il restore scarica da Google Drive il run sintetico
raccomandato, verifica file ID, dimensione e configurazione, accetta solo il
`model.ckpt` finale e non esegue fit o calibrazione sui frame reali. PatchCore
resta locale perché nel Drive accessibile non è presente un checkpoint finale.
La sezione finale esegue inferenza frozen con batch 1 su tutti i 20 frame,
calcola Image AUROC/AP, Pixel AUROC/AP e AUPRO, salva score e curva PRO e
mostra per ogni immagine crop, anomaly map e overlay. Le metriche pixel sono
sul crop intero perché il pilot non dispone di una maschera completa della ruota.
Il relativo loader autorevole è
`src/anomaly_detection/data/real_wheel_dataset.py`.

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
