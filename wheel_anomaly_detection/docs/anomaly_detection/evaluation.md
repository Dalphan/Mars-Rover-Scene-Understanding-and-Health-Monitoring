# Evaluation contract

La valutazione salva soltanto quattro metriche di ranking, senza scegliere per
ora una soglia operativa:

- image AUROC e image Average Precision;
- pixel AUROC e pixel Average Precision.

Average Precision è mantenuta perché le anomalie occupano pochi pixel e rende
visibile lo sbilanciamento che AUROC può nascondere. Gli score image-level sono
conservati tutti e producono AUROC/AP esatte, inclusa la gestione dei pareggi.
Le sole metriche pixel usano un istogramma configurabile a memoria costante
(`2048` bin di default), quindi sono approssimate ma non accumulano tutte le
mappe del dataset in RAM.

Il contratto comune richiede score in `[0, 1]`. PatchCore stima separatamente la
scala image e pixel tramite il quantile configurato delle distanze su un numero
limitato di batch clean del train. Applica poi la trasformazione monotona
`distance / (distance + scale)`: non usa statistiche o label di validation/test
e non modifica il ranking richiesto da AUROC/AP.

Per default la valutazione considera l'intera immagine. Limitarsi alla target
wheel mask è configurabile, ma è corretto soltanto se una mask equivalente è
disponibile anche in deployment; usare la ground truth soltanto nel test
renderebbe il protocollo irrealistico.

La valutazione qualitativa salva inoltre `validation_examples.png` e
`test_examples.png`. Ogni riga mostra input denormalizzato, maschera reale,
anomaly map continua con scala fissa `[0, 1]` e overlay; il contorno ciano indica
la ruota e quello verde il foro annotato. Gli esempi clean e anomali sono
selezionati con quote separate e configurabili, evitando che la prevalenza
dello split nasconda una delle due classi. Non viene mostrata una maschera
predetta binaria finché non sarà definita una soglia sulla validation.

La valutazione qualitativa salva inoltre `validation_examples.png` e
`test_examples.png`. Ogni riga mostra input denormalizzato, maschera reale,
anomaly map continua con scala fissa `[0, 1]` e overlay; il contorno ciano indica
la ruota e quello verde il foro annotato. Gli esempi clean e anomali sono
selezionati con quote separate e configurabili, evitando che la prevalenza
dello split nasconda una delle due classi. Non viene mostrata una maschera
predetta binaria finché non sarà definita una soglia sulla validation.

Il notebook salva `model.ckpt` e `metrics.json` in una directory di run Kaggle;
il JSON contiene separatamente le quattro metriche per validation e test. Le
figure qualitative sono salvate nella stessa directory e incluse tra gli
artefatti opzionalmente caricati su Google Drive.
Le figure sono incluse tra gli artefatti opzionalmente caricati su Google
Drive. Il checkpoint schema v2 incorpora la configurazione comportamentale del
modello e la verifica in modo stretto al caricamento. I metadata di run
registrano inoltre preprocessing, provenienza dei pesi, contratto di
valutazione, conteggi attesi e SHA-256 di `samples.csv`; il manifest salva gli
hash di checkpoint e metriche. `pretrained` resta informazione di provenienza,
non un vincolo di load, perché i pesi effettivi sono già nello `state_dict`.
L'upload Google Drive è opzionale e replica il meccanismo del branch
`feature/ptq-qat`: usa i Kaggle Secrets `GDRIVE_CLIENT_ID`,
`GDRIVE_CLIENT_SECRET`, `GDRIVE_REFRESH_TOKEN` e `GDRIVE_FOLDER_ID`, senza
inserire credenziali nel notebook.


## Diagnostiche estese

La classe delle metriche principali mantiene il nome `AnomalyMetrics` e il suo
contratto delle quattro metriche globali. In parallelo, la pipeline raccoglie
diagnostiche a memoria limitata nello stesso passaggio di inferenza:

- distribuzione degli image score PatchCore raw per `clean` e `hole`, con CSV,
  statistiche riassuntive e istogramma;
- clean con score raw più alto e hole con score raw più basso, salvati come
  pannelli qualitativi ordinati;
- Image AUROC e image Average Precision esatte per `severity`,
  `camera_pose`, `lighting` e `wear`, con conteggi clean/anomali espliciti;
- pixel Average Precision per gli stessi sottogruppi;
- frazione reale di pixel anomali, riportata anche come baseline casuale della
  pixel AP;
- curva PRO per componente connessa e AUPRO normalizzata fino a FPR `0.05`,
  `0.10` e `0.30`.

Per il raggruppamento per severità, che nelle righe clean è intenzionalmente
vuota, la clean eredita la severità della corrispondente hole tramite `pair_id`.
In questo modo ogni gruppo contiene sia pixel positivi sia negativi e la pixel
AP rimane definita. Le metriche image-level sono pubblicate sotto
`image_metrics_by_group`; se un sottogruppo contiene una sola classe, AUROC e
AP valgono `null` e i conteggi rendono esplicita la causa. La PRO usa
connettività a 4 pixel e non introduce una soglia operativa: integra una curva
su soglie multiple. `pro.aupro_by_max_fpr` contiene i tre cutoff, mentre
`pro.aupro` e `pro.max_fpr` restano disponibili per retrocompatibilità. I
parametri sono sotto `evaluation.diagnostics` nella configurazione Hydra.

Per ogni split vengono prodotti `*_diagnostics.json`, `*_image_scores.csv`,
`*_pro_curve.csv`, `*_raw_score_distribution.png`,
`*_highest_clean_scores.png` e `*_lowest_hole_scores.png`.
