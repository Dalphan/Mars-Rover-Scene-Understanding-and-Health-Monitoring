# Evaluation contract

La valutazione salva quattro metriche di ranking principali, senza scegliere per
ora una soglia operativa:

- image AUROC e image Average Precision;
- pixel AUROC e pixel Average Precision.

Average Precision rimane importante per lo sbilanciamento dei pixel. Gli score
image-level raw sono conservati tutti e producono AUROC/AP esatte, inclusa la
gestione dei pareggi. Le metriche pixel usano un istogramma a memoria costante
(`2048` bin di default), quindi sono approssimate.

Le anomaly map del contratto comune restano in `[0, 1]`, mentre le metriche
image-level usano lo score raw del modello. Questo evita pareggi artificiali
quando una normalizzazione limitata satura a `0` o `1`; il JSON riporta anche
conteggi e frazione degli image score normalizzati ai due estremi.

PatchCore usa `distance / (distance + scale)`. EfficientAD usa invece
`0.5 + atan(10 * calibrated_map) / pi`: la trasformazione monotona mantiene la
mappa in `[0, 1]` senza introdurre i plateau del clipping. Le metriche
image-level conservano in entrambi i casi il ranking raw.

La variante EfficientAD pose-conditioned apprende una ROI e quantili per
posizione da un calibration split clean disgiunto dal training. La validation
seleziona fra score massimo globale, massimo spaziale e media top-k spaziale;
la scelta e i risultati di ogni candidato sono salvati in
`fit_summary.image_score_ablation`. Nessun candidato viene selezionato sul test.

La diagnostica `evaluation.efficientad_calibration_ablation` riusa lo stesso
checkpoint e confronta `global`, `global_roi` e `spatial_roi` su validation e
test. In tutti e tre i casi lo score immagine e il massimo: il confronto separa
l'effetto della ROI da quello dei quantili spaziali senza il confondente top-k.
Salva metriche image/pixel/target-wheel e AUPRO in
`efficientad_calibration_ablation.json`; il test e solo descrittivo e non viene
usato per scegliere una modalita. Gli override sono transitori e lo scoring
selezionato dal fit viene sempre ripristinato.

L'ablation `evaluation.efficientad_global_topk_ablation` mantiene calibrazione
globale e nessuna ROI. Sullo stesso checkpoint confronta il massimo della mappa
ridimensionata, il massimo della mappa nativa e le medie dei migliori 2, 4, 8 e
16 pixel nativi. Sceglie il candidato solo per Image AUROC di validation e
valuta sul test esclusivamente il vincitore, salvando
`efficientad_global_topk_ablation.json`. Le anomaly map non cambiano: un
controllo interrompe l'ablation se metriche pixel, target-wheel o AUPRO variano
oltre `1e-8`. Il costo e sei passaggi di validation e uno di test, senza
retraining; al termine gli override transitori vengono ripristinati.

Con `RESTRICT_PIXELS_TO_TARGET_MASK=False` le metriche `pixel_*` principali
considerano l'intera immagine. La pipeline calcola sempre anche
`target_wheel_pixel_*` come audit separato. La target-wheel mask deriva dal
membro clean ed è condivisa dalla coppia clean/hole; non coincide con la
ground truth dell'anomalia. Limitare la metrica primaria alla ruota è corretto
soltanto se una mask equivalente è disponibile in deployment.

La valutazione qualitativa salva inoltre `validation_examples.png` e
`test_examples.png`. Ogni riga mostra input denormalizzato, maschera reale,
anomaly map continua con scala fissa `[0, 1]` e overlay; il contorno ciano indica
la ruota e quello verde il foro annotato. Gli esempi clean e anomali sono
selezionati con quote separate e configurabili, evitando che la prevalenza
dello split nasconda una delle due classi. Non viene mostrata una maschera
predetta binaria finché non sarà definita una soglia sulla validation.

Il notebook salva `model.ckpt` e `metrics.json` in una directory di run Kaggle;
il JSON contiene metriche principali, audit sulla target wheel e diagnostiche
di saturazione per validation e test. Le figure sono salvate nella stessa
directory e incluse tra gli artefatti opzionalmente caricati su Google Drive.
Per TinyGLASS, l'ablation post-hoc configurata in
`evaluation.image_score_aggregation` confronta `max`, medie top-k e quantili
sulle stesse patch del checkpoint. La variante viene scelta esclusivamente
sull'Image AUROC di validation e valutata una sola volta sul test; il risultato
è salvato in `tinyglass_image_score_aggregation.json`. Mappe e metriche pixel
non vengono modificate.

La diagnostica TinyGLASS evaluation.gaussian_sigma_ablation esegue inoltre,
sullo stesso checkpoint e sul solo split di validation, lo sweep post-hoc
sigma = 0, 1, 2, 4. Non seleziona automaticamente un sigma e non interroga il
test: salva in tinyglass_gaussian_sigma_ablation.json Pixel AUROC/AP globale e
sulla target wheel, AUPRO e AUPRO ai cutoff FPR configurati. Image AUROC/AP
devono restare uguali alla valutazione standard entro 1e-8; una variazione
interrompe il diagnostico perché indicherebbe che lo smoothing sta alterando
anche lo score immagine. Il sigma configurato viene ripristinato in un blocco
finally, anche se un candidato fallisce. Il costo è quattro passaggi aggiuntivi
di validation, ma nessun retraining.
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


## Diagnostiche estese

La classe `AnomalyMetrics` espone le quattro metriche globali, le due metriche
pixel sulla target wheel e le diagnostiche di saturazione. In parallelo, la
pipeline raccoglie diagnostiche a memoria limitata nello stesso passaggio di inferenza:

- distribuzione degli image score raw del modello per `clean` e `hole`, con CSV,
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
