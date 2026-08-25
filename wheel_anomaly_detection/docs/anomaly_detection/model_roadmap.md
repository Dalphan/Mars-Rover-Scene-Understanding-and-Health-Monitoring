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
`fixed_training_duration: true` è il default per SuperSimpleNet e TinyGLASS:
non viene eseguita model selection sulla validation e il modello usa sempre
l'ultimo stato dopo il numero configurato di epoche. Impostandolo a `false`, la
validation image AUROC seleziona invece il best state agli intervalli indicati.

- `model=efficientad_s`: implementazione full a 70.000 step con teacher/student
  PDN-S, autoencoder del paper, hard-feature mining, penalty su ImageNet-1k train
  e calibrazione separata q90/q99.5 sui soli clean di validation. Il notebook
  scarica `teacher_small.pth` dal commit fissato di `nelson1425`; l'esecuzione
  locale richiede `model.teacher_weights_path` e `HF_TOKEN`. Prima del fit vengono
  scaricati 18 shard Parquet deterministici nella cache locale; il numero reale
  di righe viene verificato rispetto ai 70.000 step e il training non effettua
  letture remote. La scelta evita i timeout ma limita il penalty dataset al
  sottoinsieme fissato dal seed. Checkpoint atomici ogni 1.000 step consentono
  il resume con `resume=true resume_run_dir=...`.
- `model=supersimplenet`: port dell'implementazione JIMS ufficiale in modalità
  unsupervised/MVTec. Usa `wide_resnet50_2` con weights torchvision
  `IMAGENET1K_V1`, feature `layer2/layer3`, adaptor, generator Perlin + rumore,
  segmentor e decision head con loss focal e truncated L1. Il preset completo è
  di 300 epoche.
- `model=tinyglass`: port del percorso TinyGLASS a griglia compatta con
  ResNet-18 torchvision `IMAGENET1K_V1`, embedding 128 canali, LAS da DTD,
  GAS con gradient ascent, hard mining e proiezione sull'ipersfera. Il preset
  completo usa 640 epoche e al massimo 392 campioni per epoca. In locale va
  impostato `model.texture_root` a `dtd/images`; il notebook scarica ed estrae
  automaticamente DTD R1.0.1 e verifica la presenza di 5.640 immagini.

Non esistono preset `*_smoke`: i test riducono direttamente epoche, dimensione
input e numero di step senza esporre configurazioni sperimentali ambigue.

### Trade-off e deviazioni dichiarate

I port mantengono pipeline specifiche dei repository ufficiali e weights
pretrained espliciti, ma non copiano l'infrastruttura di logging o export. Per
TinyGLASS il centro delle feature viene calcolato una volta: il backbone è
congelato e il port non usa una preprojection trainabile, quindi ricalcolarlo a
ogni epoca darebbe lo stesso
risultato con un costo molto maggiore. `to_deployment_module()` esporta il
modulo PyTorch della heatmap; ONNX, PTQ e conversione IMX500 restano fuori
scope. Prima dell'uso in tesi vanno registrati runtime, VRAM e QA delle heatmap.

Confronto a `256x256`:

```powershell
python scripts/anomaly_detection/run_experiment.py `
  model=patchcore_256 `
  dataset.root="D:/datasets/curiosity_wheel_hole_v1_10000"
```
