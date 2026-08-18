# Assemblaggio finale del dataset v1

Stato: completato e verificato il 18 agosto 2026.

## Risultato

Il dataset canonico e disponibile in:

```text
outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000/
```

La directory contiene una vista ordinata dei raster prodotti dai run, senza
duplicarne i byte. Gli artefatti sono materializzati tramite hardlink: il file
nella cartella dataset e il file nel run sorgente sono due nomi dello stesso
contenuto sul disco.

```text
curiosity_wheel_hole_v1_10000/
├── dataset.json
├── image_manifest.jsonl
├── images/
│   ├── train/clean/
│   ├── validation/{clean,hole}/
│   └── test/{clean,hole}/
├── masks/
│   ├── target_wheel/<split>/<condition>/
│   └── anomaly/{validation,test}/hole/
├── pool/manifest.jsonl
├── splits/{train,validation,test}.jsonl
└── planning/bulk_v1/
```

## Conteggi verificati

| Split | Clean | Hole | Totale immagini |
|---|---:|---:|---:|
| train | 7.000 | 0 | 7.000 |
| validation | 750 | 250 | 1.000 |
| test | 1.000 | 1.000 | 2.000 |
| totale | 8.750 | 1.250 | 10.000 |

Sono presenti inoltre:

- 10.000 RGB;
- 10.000 target-wheel mask;
- 1.250 anomaly mask;
- 8.750 righe nel manifest delle unita;
- 10.000 righe nel manifest per immagine;
- 7.000/1.000/2.000 righe negli indici train/validation/test.

L'assemblatore ha verificato 20.000 artefatti sorgente tramite SHA-256 prima
della pubblicazione. Il riepilogo e gli hash dei manifest e degli split sono
registrati in `dataset.json`.

## Materializzazione e sicurezza

La configurazione autorevole e
`configs/blender/final_dataset_assembly_v1.json`; l'entrypoint host e
`scripts/host/assemble_final_dataset.py`.

La procedura e fail-closed:

1. legge soltanto run completi e validati;
2. riconcilia ogni unita con il piano bulk;
3. verifica gli SHA-256 delle sorgenti;
4. costruisce la vista in staging;
5. pubblica manifest e directory soltanto dopo tutti i controlli.

Non modificare mai un PNG in place dentro la vista finale: essendo un
hardlink, la modifica altererebbe anche il raster del run sorgente. Per
trasformazioni, resize o export MVTec/Anomalib creare sempre un nuovo file in
una directory derivata. Eliminare un singolo nome hardlink non elimina gli
altri nomi dello stesso contenuto, ma la cancellazione della vista canonica va
comunque trattata come operazione distruttiva.

## Nota sui pair-lock storici

Per 1.225 coppie il contratto fotometrico produttivo e cambiato dopo la
materializzazione del piano iniziale. L'assemblatore conserva il
`pair_lock_id` effettivo e validato dal run e registra anche il valore
pianificato come `planned_pair_lock_id`. Split, sample index e categorie
semantiche restano quelli fissati dal piano; nessuna coppia viene separata.

## Pulizia post-assemblaggio

Il 18 agosto 2026 e stata eseguita una pulizia conservativa degli output. Sono
stati rimossi soltanto artefatti rigenerabili o sostituiti:

- raster dei benchmark clean e paired;
- dipendenze Python generate localmente sotto `outputs/`;
- smoke clean v1, sostituito dallo smoke v2 con gate di contatto terreno;
- revisioni visive intermedie di terreno, foro passante, bordo e profondita
  pareti, ormai incorporate nel pilot T3 canonico;
- PID terminati e log dei tre tentativi paired falliti antecedenti al run
  fotometrico diagnostico.

Sono stati conservati i run bulk e pre-bulk: oltre ai raster collegati tramite
hardlink, contengono la provenienza completa, i manifest e i metadati di
rendering. Sono stati conservati anche scena sorgente, pilot T3 canonico e
probe di drift. Dopo la pulizia sono stati ricontrollati l'hash di
`dataset.json` e i conteggi di 10.000 RGB, 10.000 target-wheel mask e 1.250
anomaly mask.

I sorgenti e le configurazioni dei benchmark rimangono nel repository: sono
piccoli e servono a riprodurre le misure documentate, anche se i raster
benchmark non vengono conservati.
