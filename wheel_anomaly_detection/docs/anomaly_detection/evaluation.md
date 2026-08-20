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

Il contratto comune richiede score in `[0, 1]`. La calibrazione necessaria per
portare distanze o logits in tale intervallo dovrà usare soltanto train o
validation, mai statistiche o label del test, per evitare data leakage.

Per default la valutazione considera l'intera immagine. Limitarsi alla target
wheel mask è configurabile, ma è corretto soltanto se una mask equivalente è
disponibile anche in deployment; usare la ground truth soltanto nel test
renderebbe il protocollo irrealistico.

Il notebook salva `model.ckpt` e `metrics.json` in una directory di run Kaggle.
Il checkpoint schema v2 incorpora la configurazione comportamentale del
modello e la verifica in modo stretto al caricamento. I metadata di run
registrano inoltre preprocessing, provenienza dei pesi, contratto di
valutazione, conteggi attesi e SHA-256 di `samples.csv`; il manifest salva gli
hash di checkpoint e metriche. `pretrained` resta informazione di provenienza,
non un vincolo di load, perché i pesi effettivi sono già nello `state_dict`.
L'upload Google Drive è opzionale e replica il meccanismo del branch
`feature/ptq-qat`: usa i Kaggle Secrets `GDRIVE_CLIENT_ID`,
`GDRIVE_CLIENT_SECRET`, `GDRIVE_REFRESH_TOKEN` e `GDRIVE_FOLDER_ID`, senza
inserire credenziali nel notebook.
