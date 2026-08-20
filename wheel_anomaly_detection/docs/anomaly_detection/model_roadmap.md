# Model roadmap

I primi quattro modelli selezionati per il progetto sono, in ordine:

1. PatchCore;
2. EfficientAD-S;
3. SuperSimpleNet;
4. TinyGLASS.

PaSTe e PaDiM-Lite restano confronti opzionali per la fase di deployment.

## PatchCore

Stato: struttura del modello implementata.

La configurazione Hydra è in
`configs/anomaly_detection/model/patchcore.yaml`. Il default usa ResNet-18 per
contenere il costo del backbone; `wide_resnet50_2` resta selezionabile come
riferimento più vicino al paper originale.

La milestone corrente estrae e aggrega le feature locali di `layer2` e
`layer3`. Non costruisce ancora la memory bank e non produce anomaly score o
heatmap. `coreset_sampling_ratio` e `num_neighbors` sono già configurati, ma
saranno utilizzati soltanto nella prossima milestone.

PatchCore non aggiorna pesi con gradient descent: backbone e BatchNorm restano
congelati, `optimizer.name` e `scheduler.name` devono quindi essere `none`.
Inventare un optimizer in questa fase non avrebbe alcun effetto sul modello.

Il baseline usa temporaneamente un resize diretto a `256x256`. La deformazione
del rapporto `4:3`, il crop/ROI della ruota e il compromesso tra risoluzione e
memoria della patch bank saranno misurati in una milestone successiva.

## Interfaccia comune

Ogni modello implementa `fit(train_loader, device=...)`, `predict(images)`,
`save(path)` e `load(path)`. `predict` restituisce sempre `AnomalyPrediction`:
uno score immagine `[B]` e una anomaly map `[B, 1, H, W]`, entrambi finiti e
normalizzati in `[0, 1]`. `forward` resta un dettaglio interno del modello.

PatchCore non implementa ancora `fit` e `predict`: la milestone corrente espone
il contratto senza simulare una memory bank o risultati non ancora disponibili.

I checkpoint registrano e validano `backbone`, layer, rapporto coreset, numero
di vicini e pooling locale. In questo modo una memory bank non può essere
ricaricata silenziosamente con un estrattore o uno scorer incompatibile.

## Rischio di scala da verificare in `fit`

Con `7000` immagini a `256x256`, la griglia `layer2` di ResNet-18 produce circa
`7,2` milioni di patch prima del coreset. Conservare il `10%` delle embedding
concatenate (`384` valori `float32`) richiederebbe circa `1,1 GB`, escluso
l'indice nearest-neighbour. Il backbone leggero da solo non rende quindi
PatchCore adatto al rover. L'implementazione di `fit` dovrà evitare di accumulare
tutte le feature sulla GPU, imporre un limite esplicito alla memory bank e
misurare il trade-off tra coreset, accuratezza, RAM e latenza.

Smoke test della sola struttura, senza scaricare i pesi:

```powershell
python scripts/anomaly_detection/inspect_model.py model.pretrained=false
```
