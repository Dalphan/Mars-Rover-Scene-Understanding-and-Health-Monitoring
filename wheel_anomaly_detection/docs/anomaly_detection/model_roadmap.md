# Model roadmap

I primi quattro modelli selezionati per il progetto sono, in ordine:

1. PatchCore;
2. EfficientAD-S;
3. SuperSimpleNet;
4. TinyGLASS.

PaSTe e PaDiM-Lite restano confronti opzionali per la fase di deployment.

## PatchCore

Stato: baseline end-to-end implementato.

La base Hydra è in `configs/anomaly_detection/model/patchcore.yaml`. Il preset
`model=patchcore_light` usa ResNet-18, frame `384x512` ed embedding finale da
`384`; `model=patchcore_reference` usa WideResNet-50-2, geometria IM224 ed
embedding `1024 -> 1024` come il baseline Amazon.

Il modello estrae patch locali sovrapposte da `layer2` e `layer3`, allinea le
griglie preservando gli assi interni della patch, applica il MeanMapper per
layer e l'aggregazione alla dimensione target. Successivamente costruisce
un reservoir uniforme a memoria limitata, applica un coreset k-center greedy su
proiezione casuale e conserva la memory bank risultante. `predict` usa distanze
1-NN per la heatmap e il reweighting PatchCore dei vicini della patch con score
massimo per lo score immagine. La heatmap viene riportata alla risoluzione di
input e filtrata con Gaussian blur.

PatchCore non aggiorna pesi con gradient descent: backbone e BatchNorm restano
congelati, `optimizer.name` e `scheduler.name` devono quindi essere `none`.
Inventare un optimizer in questa fase non avrebbe alcun effetto sul modello.

Il preset light usa resize diretto `384x512` senza deformare il rapporto `4:3`.
Il reference usa `Resize(shorter_side=256) + CenterCrop(224)` e normalizzazione
ImageNet. `model=patchcore_256` resta disponibile come confronto storico.

## Interfaccia comune

Ogni modello implementa `fit(train_loader, device=...)`, `predict(images)`,
`save(path)` e `load(path)`. `predict` restituisce sempre `AnomalyPrediction`:
uno score immagine `[B]` e una anomaly map `[B, 1, H, W]`, entrambi finiti e
normalizzati in `[0, 1]`. `forward` resta un dettaglio interno del modello.

I checkpoint registrano e validano tutti i parametri che determinano estrazione,
sampling, coreset, scoring e calibrazione. In questo modo una memory bank non
può essere ricaricata silenziosamente con un estrattore o scorer incompatibile.

## Rischio di scala da verificare in `fit`

Con `7000` immagini a `384x512`, la griglia `layer2` di ResNet-18 produce circa
`21,5` milioni di patch, cioè circa `33 GB` di embedding `float32` da 384
dimensioni prima del coreset. Il default campiona al massimo `128` patch per immagine,
mantiene un reservoir uniforme di `50000` embedding e limita la bank finale a
`2048` elementi, circa `3,1 MB` per le sole embedding. Questa scelta rende il
fit eseguibile, ma è un'approssimazione rispetto al coreset sull'intero dataset.
Il nearest-neighbour esatto resta inoltre il principale rischio di latenza sul
rover e dovrà essere misurato prima di considerare il baseline deployable.

Esecuzione completa da terminale:

```powershell
python scripts/anomaly_detection/run_experiment.py `
  dataset.root="D:/datasets/curiosity_wheel_hole_v1_10000"
```

## Modelli post-PatchCore

Stato: port dedicati e end-to-end disponibili.

I tre modelli condividono output, runner, metriche e formato checkpoint, ma non
condividono più architettura o training loop. Il train resta clean-only.
I modelli trainabili salvano checkpoint atomici nella directory del run:
`efficientad_training.ckpt`, `supersimplenet_training.ckpt` e
`tinyglass_training.ckpt`. SuperSimpleNet e TinyGLASS salvano a fine epoca con
intervallo configurabile tramite `checkpoint_interval`; il resume riparte
dall'ultima epoca completata ripristinando modello trainabile, optimizer,
scheduler quando presente, best state, contatori, RNG e generatore del
DataLoader. Un arresto durante un'epoca può quindi richiedere di ripetere solo
quell'epoca.
`fixed_training_duration: true` resta il default di SuperSimpleNet: non viene
eseguita model selection sulla validation e il modello usa sempre l'ultimo
stato dopo il numero configurato di epoche. TinyGLASS usa invece
`fixed_training_duration: false`: la validation image AUROC seleziona il best
state agli intervalli indicati.

- `model=efficientad_s`: implementazione con teacher/student PDN-S,
  autoencoder del paper, hard-feature mining e penalty su ImageNet-1k train.
  Il preset globale corrente esegue 70.000 step fissi con uno scheduler
  temporale che riduce il learning rate al 95% del training. Usa tutte le clean
  di training nell'ottimizzazione; la clean validation entra soltanto dopo il
  training per i quantili globali q90/q99.5. Lo score e `global_max`, senza ROI
  statica, quantili spaziali o selezione top-k. Il preset
  hard-quantile `balanced=0.995` e il default sperimentale; restano disponibili
  `official=0.999` e `broad=0.99`.
  Il preset operativo usa input `384x384`: teacher e student PDN producono
  mappe native `96x96` e la geometria del decoder autoencoder viene scalata
  dal riferimento `64x64` usato a `256x256`. I pesi del teacher restano
  invariati perch� il PDN e completamente convoluzionale. Questa estensione
  aumenta la risoluzione disponibile per le anomalie piccole ma non coincide
  con la geometria ufficiale EfficientAD a `256x256`; richiede un nuovo
  training e non pu� riprendere un checkpoint 256.
  Per contenere il costo della variante 384, i forward PDN senza stato di batch
  sono concatenati e le convoluzioni usano FP16 con gradient scaling su CUDA;
  quantile, hard mining e loss restano FP32. Il resume include lo stato del
  `GradScaler` e i checkpoint periodici vengono scritti ogni 5.000 step.
  Il preprocessing EfficientAD applica opzionalmente crop quadrati fissi
  selezionati sulle sole target-wheel mask clean di validation: A
  `(0,150,900,900)`, C1 `(0,148,900,900)`, C2
  `(0,153,900,900)`; D resta invariata. Le coordinate sono
  `(top,left,height,width)`. Se una coordinata porterebbe il quadrato fuori
  frame, il crop viene traslato verso l'interno e appoggiato al bordo coinvolto,
  mantenendo la dimensione `900x900`; non viene applicato padding. Un crop
  piu grande del frame resta invece invalido. I crop conservano tutte le anomaly
  mask di validation e
  test; sulle target mask clean di validation conservano il 100% dei pixel in
  A e in media il 99,6% in C1/C2. Poiche la geometria e selezionata sulla
  clean validation, tale split resta uno split di tuning: il confronto finale
  indipendente va letto sul test e il crop non va riottimizzato sul test.
  Con `spatial_calibration_enabled=true`, il 20% delle clean di training
  (almeno 32 immagini) viene selezionato deterministicamente per `image_id` e
  seed, escluso dall'ottimizzazione e processato senza augmentation. Su questo
  sottoinsieme vengono calcolati sia i quantili globali q90/q99.5 sia quantili
  per posizione sulle mappe PDN native. I quantili spaziali sono smussati,
  la scala locale ha un floor pari al 25% della scala globale e una ROI statica
  conserva i pixel presenti in almeno il 50% delle target-wheel mask clean.
  Questo evita il leakage delle coppie clean/hole di validation e test.
  La validation confronta `global_max`, `spatial_max` e top-k mean sulle
  frazioni ROI `0.05%`, `0.1%`, `0.25%` e `0.5%`; seleziona una sola variante
  tramite image AUROC e il test viene valutato una volta con tale scelta.
  Una successiva ablation diagnostica, senza model selection, mantiene invece
  `max` fisso e misura separatamente calibrazione globale senza ROI,
  calibrazione globale con ROI e calibrazione spaziale con ROI. Le tre modalita
  usano lo stesso checkpoint e non modificano il formato v4.
  Questa pipeline va dichiarata come variante pose-conditioned di EfficientAD,
  non come implementazione ufficiale pura. Il filtro `dataset.camera_poses` usa `null` come
  default per includere tutte le pose; una lista limita coerentemente train,
  clean validation, validation diagnostica e test. Il notebook e attualmente
  predisposto con `CAMERA_POSES=("A_overhead",)`: verifica i conteggi
  2.800 train, 392 validation e 808 test prima di avviare il fit. Il notebook
  scarica `teacher_small.pth` dal commit fissato di `nelson1425`; l'esecuzione
  locale richiede `model.teacher_weights_path` e `HF_TOKEN`. Prima del fit vengono
  scaricati 18 shard Parquet deterministici nella cache locale; il numero reale
  di righe viene verificato rispetto ai 70.000 step e il training non effettua
  letture remote. La scelta evita i timeout ma limita il penalty dataset al
  sottoinsieme fissato dal seed. Checkpoint atomici ogni 1.000 step consentono
  il resume con `resume=true resume_run_dir=...`. L'estensione usa il formato
  `efficientad_training_v4_spatial_calibration`; i training checkpoint
  precedenti richiedono un nuovo run.
- `model=supersimplenet`: port dell'implementazione JIMS ufficiale in modalità
  unsupervised/MVTec. Usa `wide_resnet50_2` con weights torchvision
  `IMAGENET1K_V1`, feature `layer2/layer3`, adaptor, generator Perlin + rumore,
  segmentor e decision head con loss focal e truncated L1. Il preset completo è
  di 300 epoche.
- `model=tinyglass`: port del percorso TinyGLASS a griglia compatta con
  ResNet-18 torchvision `IMAGENET1K_V1`, embedding 128 canali, LAS configurabile,
  GAS con gradient ascent, hard mining e proiezione sull'ipersfera. Il preset
  completo usa 640 epoche e al massimo 392 campioni per epoca. In locale va
  impostato `model.texture_root` a `dtd/images` per le modalita `texture` e
  `mixed`; il notebook scarica DTD solo in questi due casi. La mask
  LAS viene ridotta alla griglia feature con max pooling come nell'upstream, in
  modo che una patch resti anomala se contiene almeno un pixel sintetico.
  Il discriminatore valuta feature normali e GAS con un singolo forward
  concatenato, così la BatchNorm osserva la distribuzione congiunta usata dal
  riferimento invece di aggiornarsi separatamente sui due domini.
  `las_restrict_to_target_mask: true` interseca ogni mask Perlin con la
  target-wheel mask del train: l'augmentazione resta quindi confinata alla
  ruota e non richiede alcuna mask durante l'inferenza.
  `las_mode` accetta `texture`, `hole` e `mixed`. `texture` conserva
  DTD+Perlin; `hole` genera una sola regione irregolare connessa e interamente
  contenuta nella target wheel; `mixed` sceglie per immagine con probabilita
  `las_hole_probability`. La cavita hole viene composta nello spazio RGB con
  fattore di luminanza `las_hole_luminance_range`, rumore a bassa frequenza e
  un rim interno. `las_hole_severity_weights` configura e normalizza i pesi
  small/medium/large. Tutti i pixel modificati appartengono alla mask LAS.
  Il default `freeze_backbone: true` conserva il backbone congelato in eval e
  permette di calcolare le feature della validation una sola volta, mantenendole
  a chunk in RAM CPU. Con `freeze_backbone: false` il backbone viene ottimizzato
  in train mode con il gruppo LR separato `backbone_learning_rate`; la cache
  viene disabilitata automaticamente con warning e ogni validation ricalcola le
  feature correnti. Il preset corrente usa `learning_rate: 5e-5` (LR effettivo
  del discriminatore `1e-4`) e arresta il training dopo 40 controlli
  senza un miglioramento AUROC superiore a `early_stopping_min_delta`, salva
  immediatamente il checkpoint e ripristina il best state.
  `early_stopping_patience: null` disabilita l'arresto. Cache ed early stopping
  riducono il runtime, ma la cache richiede circa 125 MiB per 1.000 immagini
  256x256 e la selezione continua a dipendere dalle anomalie del solo split di
  validation, mai dal test.
  `freeze_backbone` e `backbone_learning_rate` fanno parte della configurazione
  verificata dai checkpoint; cambiare modalità richiede quindi un nuovo run.
  Training state e best state includono anche i pesi del backbone. DTD è
  richiesto soltanto dalle modalita `texture` e `mixed` per un training o resume non ancora
  concluso: un checkpoint finale può essere costruito e caricato per inferenza
  senza mantenere il dataset di texture.

Non esistono preset `*_smoke`: i test riducono direttamente epoche, dimensione
input e numero di step senza esporre configurazioni sperimentali ambigue.

Primo confronto fine-tuning consigliato, sempre come nuovo run e con lo stesso
seed e pose crop del baseline frozen:

```powershell
python scripts/anomaly_detection/run_experiment.py `
  model=tinyglass `
  dataset.root="D:/datasets/curiosity_wheel_hole_v1_10000" `
  +model.pose_crop_enabled=true `
  model.freeze_backbone=false `
  model.backbone_learning_rate=0.00001 `
  model.epochs=20 `
  model.early_stopping_patience=7 `
  model.cache_validation_features=false
```

Gli iperparametri si scelgono sulla validation; il test viene usato soltanto
per il confronto finale con il run frozen.

Preset frozen wheel-aware sulla sola posa A:

```powershell
python scripts/anomaly_detection/run_experiment.py `
  model=tinyglass `
  +experiment=tinyglass_pose_a `
  dataset.root="D:/datasets/curiosity_wheel_hole_v1_10000"
```

Il preset applica a train, validation e test il crop fisso `A_overhead`
`(90, 240, 720, 720)`, centrato dal contratto della camera sulla ruota target,
e lo ridimensiona a 384x384.
Mantiene il backbone frozen, abilita LAS hole wheel-aware, disabilita la hypersphere
projection e sceglie il best checkpoint esclusivamente sulla validation. La
fusione usa la griglia `layer2` a 48x48, interpolando `layer3` verso l'alto,
usa `gaussian_sigma=1`, LR config `5e-5`, patience 40 e pesi di sampling
small/medium/large `70/25/5` per il nuovo esperimento mirato ai fori piccoli.
Il default retrocompatibile `50/40/10` deriva dal contratto Blender; le
dimensioni sono scalate sul diametro equivalente della ruota visibile. I
profili irregolari seguono i pesi 45/40/15% di
jagged slit, branched tear e peeled window. Queste
modifiche introducono il formato training checkpoint
`tinyglass_training_v4_configurable_las`: i checkpoint training precedenti
richiedono un nuovo run e non possono essere ripresi.

La griglia 32x32 quadruplica patch, memoria della cache di validation e lavoro
del discriminatore rispetto alla griglia 16x16. Il crop piu stretto compensa
solo il contenuto inquadrato, non questo costo computazionale.

### Trade-off e deviazioni dichiarate

I port mantengono pipeline specifiche dei repository ufficiali e weights
pretrained espliciti, ma non copiano l'infrastruttura di logging o export. Per
TinyGLASS il centro delle feature viene calcolato una volta nel percorso frozen;
se il backbone è trainabile viene invece ricalcolato prima di ogni epoca, così
centro, raggi LAS/GAS e feature restano nello stesso spazio. Il costo è una
passata clean aggiuntiva per epoca, registrata tramite `center_recomputations`
e `center_patches_total`. `to_deployment_module()` esporta il modulo PyTorch
della heatmap; ONNX, PTQ e conversione IMX500 restano fuori scope. Prima
dell'uso in tesi vanno registrati runtime, VRAM e QA delle heatmap.

La LAS hole e una surrogate 2D domain-specific e non una replica del generatore
3D: non puo riprodurre parallax, auto-occlusioni, pareti geometriche o separare
in modo affidabile tread e shoulder dalla sola target-wheel mask. La scala basata
sull'area proiettata e inoltre un'approssimazione sotto prospettiva. Il vincolo
di contenimento evita label noise ai bordi, ma puo sotto-campionare le severita
large quando la ruota e parzialmente visibile. Il pattern scuro puo confondersi
con ombre reali; per questo `texture` e `mixed` restano disponibili come
ablazioni e gli iperparametri vanno scelti solo sulla validation.

Confronto a `256x256`:

```powershell
python scripts/anomaly_detection/run_experiment.py `
  model=patchcore_256 `
  dataset.root="D:/datasets/curiosity_wheel_hole_v1_10000"
```
