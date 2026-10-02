# Pilot zero-shot su ruote reali

Stato: caricamento, crop, quattro modelli, restore Google Drive e valutazione
zero-shot implementati; l'esecuzione dei checkpoint resta da effettuare su Kaggle.

## Scopo

Il pilot usa 20 immagini MAHLI reali (10 `hole`, 10 `no_visible_hole`) come
controllo synthetic-to-real strettamente held-out. Le immagini non vanno usate
per training, calibrazione, scelta del checkpoint, scelta dello score, tuning
della soglia o selezione del crop.

Il notebook Kaggle dedicato è
`notebooks/kaggle_real_wheel_zero_shot.ipynb`. È
autosufficiente e duplica il loader autorevole in
`src/anomaly_detection/data/real_wheel_dataset.py`.

## Contratto del dataset Kaggle

La directory collegata come Kaggle Dataset deve contenere:

```text
real_samples.csv
real_crops.csv
hole_masks.csv
images/
  hole/
  no_visible_hole/
masks/
  hole/
```

Il dataset Kaggle si chiama `Real Mars Rover Wheel AD` ed è montato direttamente
in `/kaggle/input/datasets/dalphan01/real-mars-rover-wheel-ad/real_wheel_pilot_v2`; il notebook usa questo path senza
eseguire discovery ricorsiva.

La tabella versionata delle coordinate è
`configs/anomaly_detection/real_wheel_pilot_v2_crops.csv`; nel dataset Kaggle
va inclusa con nome `real_crops.csv`. La copia locale del pilot usa lo stesso
nome nella propria root.

## Contratto dei crop

- Coordinate in pixel originali, ordine `(top, left, height, width)`.
- Immagini originali: `1200 x 1632` in ordine `(height, width)`.
- I crop v2.1 sono quadrati e specifici per immagine, con lato tra 900 e 1150
  pixel; anche `top` e `left` sono definiti per singolo frame.
- Non viene applicato padding.
- Policy: `manual_wheel_geometry_tight`; versione: `real_wheel_crop_v2_1`.
- Le coordinate sono state scelte dal contorno esterno della ruota. Le maschere
  dei fori sono state usate soltanto dopo il congelamento delle coordinate per
  verificare che nessun pixel annotato fosse escluso.

Un singolo crop per Sol non è sufficiente: nel Sol 3195 posizione e
orientamento della ruota cambiano tra sequenze. La v2.1 adatta posizione e lato
per portare la ruota il più possibile in primo piano: 16 crop su 20 sono più
piccoli della v1, con lato mediano di 950 pixel.

Il notebook ricava automaticamente dimensione e normalizzazione dalla famiglia
selezionata. Il resize RGB è bilineare con antialiasing; quello della maschera è
nearest-neighbour. Le negative ricevono una maschera tutta zero.

## Selezione del modello

La variabile unica `MODEL_NAME` accetta `patchcore`, `efficientad_s`,
`supersimplenet` e `tinyglass`. I preset sono copiati dal notebook roadmap e
determinano il preprocessing senza modificare i crop nativi:

| Modello | Input | Normalizzazione nel dataset |
|---|---:|---|
| PatchCore light | 384x512 | ImageNet |
| EfficientAD-S | 256x256 | nessuna, è interna al modello |
| SuperSimpleNet | 256x256 | ImageNet |
| TinyGLASS | 384x384 | ImageNet |

Il pilot usa batch di inferenza 1. Con `CHECKPOINT_SOURCE="google_drive"`
scarica automaticamente il run raccomandato; `local` usa invece
`MODEL_CHECKPOINT_PATH` e `none` disabilita il restore. Il caricamento non
riscarica i pesi ImageNet o il teacher EfficientAD: costruisce l'architettura,
carica l'intero `state_dict` e verifica schema, famiglia, preprocessing,
configurazione e stato fitted. I checkpoint intermedi `*_training.ckpt`
vengono rifiutati.

### Folder Google Drive verificati

- `1LEskJG5sIdUBHIfws1keaaZxArvD-WWU`: EfficientAD-S global 256, 70k
  step; raccomandato per localizzazione e generalizzazione complessiva.
- `1-M2ikLWC9OnTJOm1PjBVJxgs7oVcfCk4`: EfficientAD-S spatial 256;
  Image-AUROC validation più alta ma localizzazione molto peggiore.
- `1XPHgLw2jVMiRJRtwouqtIWFrQeEHDBeA`: SuperSimpleNet 256, Perlin 0,2,
  score top 1%, anomalie sintetiche non ristrette; compromesso raccomandato.
- `1Xo5ELdDZdMKsfhKjuDT2HaQm4RqyTonb`: TinyGLASS 384/layer2;
  raccomandato e migliore nella selezione validation risoluzione/griglia.
- `1ft0zDYSQrwE9eomJ4Ksg1LmLmWr_NZt4`: TinyGLASS 384/layer3.
- `1aAs2KVGiwNZ0zfHEk_8nHSGOyp9gAxOW`: TinyGLASS 512/layer2.
- `1hWFs0qKm4xqGiT98ZH8XAJCDTl06u5iL`: TinyGLASS 512/layer3; il nome
  del folder è obsoleto e non descrive la configurazione reale.
- PatchCore: nessun `model.ckpt` finale trovato nel Drive accessibile.

Il download usa un file `.part`, verifica dimensione e file ID e solo dopo
sostituisce atomicamente la cache. La cache è indicizzata dal file ID; la
configurazione e lo `state_dict` sono comunque caricati in modalità stretta.
Impostare `FORCE_CHECKPOINT_DOWNLOAD=True` per ignorare una cache sospetta.
Servono i secret Kaggle `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET` e
`GDRIVE_REFRESH_TOKEN`.

## Inferenza, metriche e output

Il DataLoader resta congelato con batch size 1 e il modello esegue un solo
forward per immagine. Lo stesso risultato alimenta metriche, tabella degli
score e visualizzazioni, evitando un secondo passaggio potenzialmente diverso.

Le metriche riportate sono Image AUROC e Image AP sugli score raw, Pixel AUROC
e Pixel AP sulle mappe normalizzate, baseline casuale della Pixel AP e AUPRO
con FPR massimo 0,05, 0,10 e 0,30. Le metriche pixel operano sul crop intero e
sulle maschere approssimative dei fori; le metriche target-wheel non vengono
calcolate perché manca una maschera completa della ruota.

Per ogni checkpoint il notebook scrive sotto
`/kaggle/working/real_wheel_zero_shot/<model>/<checkpoint-file-id>/`:

- `metrics.json`, con protocollo e metriche aggregate;
- `image_scores.csv`, con score raw/normalizzato e statistiche della map per
  ogni immagine;
- `pro_curve.csv`;
- `predictions_page_*.png`, con crop, anomaly map e overlay per tutti i frame.

Le figure mostrano in verde il bordo dell'annotazione approssimativa, senza
usarlo nel forward del modello. Tutti gli output sono descrittivi: non possono
essere usati per cambiare checkpoint, score, smoothing, soglia o crop.

## Gate già verificati

- 20 immagini e 20 crop con ID uno-a-uno;
- 10 maschere presenti esclusivamente per le immagini `hole`;
- immagini e maschere a `1200 x 1632`;
- crop interamente nei limiti delle immagini;
- il 100% dei pixel annotati rimane nel crop nativo;
- tutte le 10 maschere restano non vuote a `384 x 384`; il minimo osservato è
  373 pixel positivi.

Questi controlli sono stati chiusi prima dell'upload. Su Kaggle il notebook si
limita a mostrare box dei crop e input esatti del modello con overlay rosso
delle maschere approssimative.

## Revisione avversariale e limiti

- Un crop manuale per immagine può introdurre bias di composizione. Per
  compensare distanza e prospettiva, la v2.1 varia il lato tra 900 e 1150 pixel
  cercando una scala apparente simile della ruota. Il lato non viene fornito al
  modello, le coordinate restano congelate prima dell'inferenza e non vengono
  corrette osservando score o heatmap; un residuo segnale spurio di scala resta
  comunque possibile.
- Le immagini dello stesso Sol sono rotazioni successive e non sono campioni
  indipendenti. Le metriche aggregate saranno solo descrittive; score e
  heatmap per immagine sono obbligatori.
- Le scale degli score raw sono specifiche della famiglia: tra modelli vanno
  confrontati ranking e metriche, non i valori assoluti degli score.
- Le maschere dei fori sono approssimative e non supportano affermazioni di
  precisione fine sulla segmentazione.
- Non esiste ancora una maschera della ruota completa per i frame reali: non si
  possono calcolare in modo coerente metriche pixel limitate alla sola ruota.
- La configurazione del notebook deve coincidere esattamente con il checkpoint.
  Un checkpoint di una diversa ablation o risoluzione viene rifiutato invece di
  essere caricato parzialmente. Il manifest dei crop nativi non cambia.
- I nomi di alcuni folder Drive sono obsoleti. La selezione usa il contenuto
  verificato di `run_manifest.json`/`metrics.json`, mai il solo nome del
  folder.
- La scelta dei run raccomandati è congelata sulle metriche sintetiche e non
  viene modificata osservando immagini, score o heatmap reali.
- Il codice dei quattro modelli è duplicato intenzionalmente dal notebook
  roadmap per mantenere Kaggle autosufficiente. Le due copie devono restare
  sincronizzate quando cambia un'architettura o il contratto checkpoint.

## Passo successivo

Eseguire su Kaggle una run separata per ciascun checkpoint congelato e archiviare
gli output senza usare il pilot reale per scegliere il vincitore.
