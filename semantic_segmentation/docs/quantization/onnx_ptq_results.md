# Studio ONNX PTQ — risultati dei modelli di segmentazione

Ultimo aggiornamento: 11 settembre 2026.

Questo documento raccoglie i risultati ottenuti nel notebook
`notebooks/kaggle_s5mars_onnx_ptq.ipynb`. I valori provengono dagli output
Kaggle condivisi durante lo studio e vanno aggiornati dopo ogni nuova variante.

## Contratto sperimentale

- Modello: U-Net con encoder MobileNetV2.
- Checkpoint FP32 su Google Drive: cartella
  `1Ub1bDNDwNMTm5NSLnyIC6DVm5IRcLCmE`.
- Input: batch 1, RGB, `1 × 3 × 512 × 512`.
- Output: logits, `1 × 9 × 512 × 512`.
- Dataset di valutazione: split `val`, 200 immagini.
- Classi: 9.
- Ignore index: `-100`.
- Benchmark: 10 warm-up e 50 esecuzioni misurate.
- Latenza: solo forward, con input già sul device.
- PyTorch: CUDA.
- ONNX Runtime: 1.26.0, `CUDAExecutionProvider` con
  `CPUExecutionProvider` come fallback.
- ONNX opset: 18.

## Riepilogo

| Variante | Dimensione | Pixel accuracy | mIoU | Media (ms) | Mediana (ms) | p95 (ms) | Immagini/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 CUDA | n.d. | 0.932901974 | 0.719812688 | 13.128 | 12.846 | 15.233 | 76.17 |
| ONNX FP32 CUDA | 25.573 MiB | 0.932901974 | 0.719812688 | 11.496 | 11.486 | 11.739 | 86.99 |
| ONNX FP16 CUDA | 12.962 MiB | 0.932940865 | 0.720057234 | 8.139 | 7.421 | 9.541 | 122.87 |

Rispetto a ONNX FP32, ONNX FP16 riduce la dimensione del 49.31% e la
latenza media di circa il 29.2%. Rispetto a PyTorch FP32, la latenza media
scende di circa il 38.0%. Le piccole variazioni positive delle metriche non
indicano un miglioramento appreso: derivano dall'arrotondamento FP16.

## IoU per classe

| Classe | PyTorch / ONNX FP32 | ONNX FP16 |
|---|---:|---:|
| Background | n.d. | n.d. |
| Bedrock | 0.783658739 | 0.783679011 |
| Hole | 0.000000000 | 0.000000000 |
| Ridge | 0.961254519 | 0.961635599 |
| Rock | 0.644567216 | 0.645053448 |
| Rover | 0.882614662 | 0.883062477 |
| Sand / Soil | 0.905156431 | 0.905160954 |
| Sky | 0.964602033 | 0.965047263 |
| Track | 0.616647901 | 0.616819122 |

`Background` è indicato come non disponibile perché non ha supporto/unione
nello split considerato. `Hole` è presente ma non ottiene veri positivi.

## Controlli degli artefatti

### ONNX FP32

- File: `/kaggle/working/onnx/smp_unet_mobilenet_v2_fp32.onnx`.
- Input e output FP32.
- Checker ONNX: superato.
- Metriche coincidenti con la baseline PyTorch.

### ONNX FP16

- File: `/kaggle/working/onnx/smp_unet_mobilenet_v2_fp16.onnx`.
- Input e output pubblici FP32; calcolo interno e 128 initializer FP16.
- Initializer FP32: 0.
- Cast di frontiera: 2.
- Checker ONNX: superato.
- Parità rispetto a ONNX FP32 su un'immagine:
  - errore assoluto massimo dei logits: 0.122989655;
  - errore assoluto medio: 0.004914411;
  - prediction agreement: 0.999900818, cioè 26 pixel differenti su 262.144;
  - `allclose=False` con `rtol=0.01` e `atol=0.01`.

`allclose=False` non è considerato un fallimento per FP16: verifica ogni
singolo logit, mentre le predizioni e le metriche complete mostrano che la
segmentazione è rimasta stabile.

## Decisioni

1. ONNX FP32 è accettato come baseline di esportazione.
2. ONNX FP16 è accettato come variante a precisione ridotta.
3. TensorRT non è necessario per FP32 o FP16: questi test usano CUDA EP.
4. INT8 sarà studiato separatamente partendo da ONNX FP32, non da FP16.
5. Gli artefatti e i benchmark restano distinti nel file
   `benchmark_results.json`.

## Calibrazione INT8 — preparazione implementata

La sezione 13 del notebook ora implementa il primo gate, senza modificare il
modello. In particolare:

1. registra se ONNX Runtime espone `TensorrtExecutionProvider`; questo dato
   da solo non dimostra ancora che il provider possa inizializzarsi;
2. usa un `CalibrationDataReader` minimale basato sul DataLoader
   esistente;
3. prepara esattamente 500 input FP32 casuali dallo split `train`, con
   seed 42, batch 1 e shape `1 × 3 × 512 × 512`;
4. non usare le maschere durante la calibrazione;
5. salva un manifesto con strategia, seed, fingerprint del dataset e indici
   originali selezionati;
6. esegue un dry-run degli input sul modello ONNX FP32 senza generare ancora
   il modello INT8.

Il codice controlla shape, tipo FP32 e valori finiti di tutti i 500 input,
unicità/intervallo degli indici e presenza per immagine di ciascuna classe;
esegue inoltre tre forward di prova e verifica il riavvolgimento del reader.
La presenza delle classi usa i metadati `class_labels`, mentre le maschere non
entrano nel calibratore.

### Risultato del gate

- Stato: `passed`.
- Strategia: `random`, seed 42.
- Campioni: 500 immagini dello split `train`.
- Fingerprint sorgente: `8480fe7c14a9dbfe`.
- Fingerprint sottoinsieme: `12bdb45e13a85706`.
- Input: `images`, FP32, `1 × 3 × 512 × 512`.
- Intervallo osservato dopo normalizzazione: da `-2.117903948` a
  `2.570283175`.
- Tutti gli input e gli output dei tre dry-run sono finiti.
- Output dei dry-run: `1 × 9 × 512 × 512`.
- `rewind()`: funzionante.
- Provider esposti da ONNX Runtime: TensorRT, CUDA e CPU.

| Classe | Immagini su 500 |
|---|---:|
| Background | 0 |
| Bedrock | 466 |
| Hole | 5 |
| Ridge | 136 |
| Rock | 267 |
| Rover | 14 |
| Sand / Soil | 486 |
| Sky | 95 |
| Track | 7 |

`Background` è assente dai metadati anche se resta una classe del modello.
`Hole` e `Track` sono invece presenti ma rare. Questi conteggi misurano la
presenza nelle immagini, non il numero di pixel: sono quindi un controllo
preliminare di rappresentatività, non una misura della distribuzione reale
degli input al calibratore.

Il fatto che TensorRT compaia nell'elenco indica che il provider è installato,
ma non ne prova ancora l'inizializzazione o l'esecuzione INT8.

Questo gate separa gli errori del dataset dagli errori della quantizzazione.
Dopo averne verificato l'output si sceglieranno formato INT8, metodo di
calibrazione e provider di esecuzione.

## Configurazione PTQ INT8 scelta

Prima configurazione implementata nella sezione 13 del notebook:

| Opzione | Valore | Motivazione |
|---|---|---|
| Metodo | Static PTQ | Percorso adatto a una CNN |
| Formato | QDQ | Rappresentazione esplicita adatta a TensorRT |
| Calibrazione | MinMax | Baseline semplice e senza clipping |
| Attivazioni | QInt8 simmetrico | INT8 signed con zero-point 0 |
| Pesi | QInt8 simmetrico | INT8 signed con zero-point 0 |
| Granularità pesi | Per-channel | Una scala per canale di output delle convoluzioni |
| Granularità attivazioni | Per-tensor | Configurazione supportata da TensorRT |
| Range ridotto | No | Usa tutto il range disponibile a 8 bit |
| Operatori | Conv | Primo test conservativo, senza quantizzare Resize/Add |
| Dati | 500 random, seed 42 | Sottoinsieme già verificato |

La cella genera
`/kaggle/working/onnx/smp_unet_mobilenet_v2_int8_qdq.onnx`, controlla il
grafo con ONNX, conta nodi Q/DQ e initializer INT8 e verifica che tutti gli
zero-point siano nulli. Non inizializza ancora TensorRT e non misura metriche
o latenza.

### Risultato della creazione QDQ

| Proprietà | Valore |
|---|---:|
| Dimensione | 7.031 MiB |
| Riduzione rispetto a ONNX FP32 | 72.51% |
| Tempo di quantizzazione | 18.528 s |
| QuantizeLinear | 119 |
| DequantizeLinear | 245 |
| Initializer INT8 | 245 |
| Initializer UINT8 | 0 |
| Zero-point non nulli | 0 |

- Checker ONNX: superato.
- Input pubblico: FP32, `1 × 3 × 512 × 512`.
- Output pubblico: FP32, `1 × 9 × 512 × 512`.
- Sottoinsieme: `random`, 500 immagini, fingerprint
  `12bdb45e13a85706`.
- Provider richiesti durante la calibrazione: CUDA con fallback CPU.

Il numero di nodi Q e DQ non deve essere uguale: i DQ comprendono anche pesi
quantizzati e tensori condivisi. Il conteggio degli initializer INT8 include
sia dati quantizzati sia costanti come gli zero-point, quindi non coincide
necessariamente con il numero di convoluzioni.

ONNX Runtime ha stampato due avvisi che consigliano il preprocessing dedicato.
Non sono errori e non invalidano il file: il modello aveva già shape statiche,
ha attraversato la shape inference interna di `quantize_static` e ha superato
il checker completo. Per mantenere questo primo percorso minimale, il modello
viene accettato come baseline. Il preprocessing separato verrà introdotto solo
se emergono problemi di copertura, esecuzione o accuratezza.

## Gate di esecuzione TensorRT

La sezione 14 del notebook crea una sessione con priorità TensorRT → CUDA → CPU
e svolge un solo forward. Il profiling registra il provider che ha realmente
eseguito operazioni, perché la sola presenza di TensorRT in
`session.get_providers()` non ne dimostra l'utilizzo.

Per il modello QDQ non viene passata una calibration table esterna: le scale
sono già contenute nel grafo. La cache TensorRT è separata usando l'hash SHA-256
dell'artefatto, così un modello modificato non riusa un engine vecchio.

### Primo tentativo TensorRT

Il primo tentativo non ha eseguito TensorRT:

- `TensorrtExecutionProvider` compariva tra i provider installati;
- il caricamento di `libonnxruntime_providers_tensorrt.so` è fallito perché
  mancava `libnvinfer.so.10`;
- ONNX Runtime è quindi ricaduto su CUDA e CPU;
- i 63 eventi `Memcpy` segnalati appartengono a questo fallback e non sono una
  misura delle prestazioni TensorRT.

Il modello QDQ non era la causa dell'errore: mancavano le librerie native. Il
notebook ora installa `tensorrt-cu12==10.9.0.34` e precarica
`libnvinfer.so.10`, `libnvinfer_plugin.so.10` e
`libnvonnxparser.so.10` prima di importare ONNX Runtime.

### Secondo tentativo TensorRT

Esito: **superato**.

- TensorRT installato: `10.9.0.34`;
- provider attivi: TensorRT, CUDA e CPU, con TensorRT in prima posizione;
- eventi di profiling: 64 TensorRT e 63 CPU;
- output finito e con shape `(1, 9, 512, 512)`;
- agreement delle predizioni rispetto a ONNX FP32: `0.98846817`;
- errore assoluto medio dei logit: `0.32215053`;
- errore assoluto massimo dei logit: `6.42917776`;
- creazione sessione: `1.812 s`;
- primo forward con engine già in cache: `13.794 ms`.

I conteggi del profiler sono **eventi**, non il numero di nodi del grafo. Il
risultato dimostra che TensorRT viene realmente usato, ma anche che l'esecuzione
è mista TensorRT/CPU. Il tempo del singolo forward del gate non viene usato come
benchmark: la cache conteneva già un engine e una sola misura non è robusta.

## Benchmark INT8 QDQ: MinMax random e class-aware

Valutazione su 200 immagini dello split validation:

| Metrica | ONNX FP32 | MinMax random | MinMax class-aware |
|---|---:|---:|---:|
| Pixel accuracy | 0.932902 | 0.906949 | 0.907202 |
| mIoU | 0.719813 | 0.636974 | 0.630926 |
| Latenza media (ms) | 11.495897 | circa 5.040453 | 5.040453 |
| Throughput (img/s) | 86.99 | circa 198.39 | 198.39 |

Differenze class-aware rispetto a random:

- pixel accuracy: +0.000253;
- mIoU: -0.006048;
- agreement sul probe: da 0.988468 a 0.989544;
- errore medio dei logit: da 0.322151 a 0.333862;
- latenza: sostanzialmente invariata.

Il class-aware migliora Ridge (0.770114 → 0.807123) e Sky
(0.721285 → 0.769618), ma peggiora soprattutto Rock
(0.537906 → 0.498989) e Rover (0.826667 → 0.747494). La qualità complessiva
non migliora: rispetto a ONNX FP32 perde 0.088887 di mIoU.

Sul piano prestazionale INT8 è circa 2.28 volte più veloce di ONNX FP32 e
circa 1.61 volte più veloce di ONNX FP16. Il costo è però una perdita di mIoU
troppo alta per accettare questa configurazione come risultato finale.

Il prossimo esperimento controllato torna alla selezione random e cambia
soltanto il metodo di calibrazione da MinMax a Entropy.

## Tentativo Entropy: limite RAM

Con 500 immagini a 512 × 512, il primo tentativo Entropy ha saturato la RAM
dentro collect_data. In ONNX Runtime 1.26 il calibratore a istogrammi espone
come output le attivazioni candidate e conserva quelle di tutti gli input
ricevuti dalla singola chiamata prima di costruire gli istogrammi.

Il notebook ora usa la suddivisione supportata da quantize_static:

- INT8_CALIBRATION_CHUNK_SIZE = 5;
- il CalibrationDataReader implementa len e set_range;
- ogni chiamata elabora cinque immagini, aggiorna gli istogrammi cumulativi e
  libera gli output intermedi;
- dataset e metodo restano gli stessi, ma il merge incrementale degli istogrammi
  può produrre soglie non identiche bit per bit rispetto al calcolo monolitico.

Un chunk pari a 1 riduce ulteriormente il picco RAM, al costo di maggiore tempo.
Il valore deve essere positivo e dividere esattamente il numero di campioni.

## Benchmark INT8 QDQ: Entropy random

Configurazione: 500 immagini random, chunk da 5, QDQ S8S8 simmetrico,
per-channel sui pesi e sole convoluzioni quantizzate.

| Metrica | ONNX FP32 | MinMax random | Entropy random |
|---|---:|---:|---:|
| Pixel accuracy | 0.932902 | 0.906949 | 0.925775 |
| mIoU | 0.719813 | 0.636974 | 0.691266 |
| Latenza media (ms) | 11.495897 | circa 5.040453 | 4.252946 |
| Latenza mediana (ms) | 11.486289 | circa 5.163211 | 5.010997 |
| P95 (ms) | 11.738532 | circa 5.233913 | 5.109452 |
| Throughput da media (img/s) | 86.99 | circa 198.39 | 235.13 |

Entropy recupera 0.054292 mIoU rispetto a MinMax random. Rimane una perdita di
0.028547 rispetto a ONNX FP32, molto più contenuta rispetto ai test MinMax.
L'agreement del probe sale a 0.996223 e l'errore medio dei logit scende a
0.215088.

La media di latenza Entropy è sensibilmente inferiore alla mediana, mentre P95
e mediana restano vicini a 5 ms. Per il confronto tra metodi è quindi più
prudente usare la mediana: non c'è ancora evidenza che il metodo di calibrazione
abbia cambiato realmente la velocità del grafo.

## Prima costruzione dell'engine TensorRT

Per ogni nuovo artefatto INT8 cambia l'hash del modello e la relativa engine
cache parte vuota. La prima creazione della sessione può impiegare diversi
minuti mentre TensorRT prova e seleziona le implementazioni dei kernel. Se
l'esecuzione viene interrotta dopo che l'engine è stato scritto, il secondo
tentativo trova la cache e termina rapidamente.

Il notebook ora:

- stampa se la engine cache è hit o miss e segnala le fasi sessione, forward e
  salvataggio profiling;
- abilita una timing cache condivisa tra varianti sullo stesso hardware;
- continua a usare una engine cache separata per hash del modello;
- salva artifact, probe e benchmark con una chiave contenente metodo e sampler,
  evitando che Percentile sovrascriva i risultati successivi.

## Benchmark INT8 QDQ: Percentile random

Configurazione: 500 immagini random, calibrazione a blocchi, QDQ S8S8
simmetrico, pesi per-channel e sole convoluzioni quantizzate.

| Metrica | ONNX FP32 | Entropy random | Percentile random |
|---|---:|---:|---:|
| Pixel accuracy | 0.932902 | 0.925775 | 0.923594 |
| mIoU | 0.719813 | 0.691266 | 0.686441 |
| Latenza media (ms) | 11.495897 | 4.252946 | 4.398065 |
| Latenza mediana (ms) | 11.486289 | 5.010997 | 4.646585 |
| P95 (ms) | 11.738532 | 5.109452 | 5.160242 |
| Throughput da media (img/s) | 86.99 | 235.13 | 227.35 |

Percentile perde 0.004825 mIoU rispetto a Entropy e 0.033372 rispetto a ONNX
FP32. Migliora Track rispetto a Entropy (0.646512 contro 0.626030), ma peggiora
le altre classi valutabili. Entropy resta quindi il miglior metodo INT8 provato
finora in termini di accuratezza.

Le latenze dei due metodi sono entrambe nell'ordine dei 5 ms. La differenza tra
media e mediana indica variabilità della misura e non giustifica la scelta di
Percentile al posto di Entropy.
## Verifica fallback TensorRT FP16

Il primo output ottenuto con Entropy e TRT_FP16_ENABLE=True è numericamente
quasi identico al precedente:

- pixel accuracy: 0.925775;
- mIoU: 0.691266;
- errore medio dei logit: 0.215088;
- latenza media: 4.252946 ms;
- latenza mediana: 5.010997 ms;
- P95: 5.109452 ms.

Questo tentativo non viene ancora considerato una prova valida del fallback
FP16. L'output mostra la vecchia artifact key e la precedente implementazione
della cache: la engine cache usava solo l'hash del modello. Poiché l'artefatto QDQ
non cambia quando si abilita TRT_FP16_ENABLE, TensorRT può aver ricaricato
l'engine costruito con FP16 disabilitato.

Il notebook ora include nella firma della cache engine cache: hash del modello,
versioni ORT/TensorRT, GPU, compute capability, device id e flag INT8/FP16.
Il prossimo probe con FP16 userà quindi obbligatoriamente una cartella distinta.

## Benchmark Entropy con operatori estesi e fallback TensorRT FP16

Configurazione del run dell'11 settembre 2026:

- calibrazione Entropy su 500 immagini random, seed 42 e chunk da 5;
- QDQ S8S8 simmetrico, pesi per-channel;
- operatori richiesti: Conv, ConvTranspose, Resize, MaxPool e AveragePool;
- TensorRT INT8 abilitato e fallback TensorRT FP16 abilitato;
- engine cache identificata anche dai flag INT8/FP16.

Il grafo FP32 contiene 63 Conv, 5 Resize, 10 Add, 35 Clip, 4 Concat e 10
Relu. ConvTranspose, MaxPool e AveragePool non sono presenti. Nel grafo
quantizzato tutte le 63 Conv risultano racchiuse da Q/DQ, mentre nessuno dei 5
Resize risulta completamente racchiuso da Q/DQ. L'ampliamento richiesto non ha
quindi aggiunto operatori non-Conv alla copertura INT8 effettiva; il run resta
utile come verifica del fallback FP16 con cache correttamente distinta.

### Artefatto e runtime

| Proprietà | Valore |
|---|---:|
| Dimensione INT8 | 7.031927 MiB |
| Riduzione rispetto a ONNX FP32 | 72.503978% |
| Tempo di quantizzazione | 1016.714 s |
| Conv racchiuse da Q/DQ | 63/63 |
| Resize racchiusi da Q/DQ | 0/5 |
| Eventi TensorRT nel probe | 64 |
| Eventi CPU nel probe | 63 |

I valori del profiler sono eventi e non conteggi di nodi. L'esecuzione resta
mista TensorRT/CPU.

### Accuratezza, parità e latenza

| Metrica | Risultato |
|---|---:|
| Pixel accuracy | 0.925775146 |
| mIoU | 0.691265541 |
| Prediction agreement con ONNX FP32 | 0.996223450 |
| Errore medio assoluto dei logit | 0.215088069 |
| Errore massimo assoluto dei logit | 3.006928444 |
| Latenza media | 4.294130 ms |
| Latenza mediana | 5.096832 ms |
| P95 | 5.186370 ms |
| Throughput da media | 232.876 immagini/s |

Accuratezza e parità coincidono, entro l'arrotondamento, con il precedente
Entropy random Conv-only. Il fallback FP16 non ha quindi prodotto una variazione
materiale dell'output. Anche la latenza resta nello stesso ordine di grandezza.

### Efficienza GPU

| Metrica | Risultato |
|---|---:|
| Potenza media campionata | 65.219950 W |
| Energia per immagine | 201.814394 mJ |
| Immagini per joule | 4.955048 |
| Memoria GPU di picco del processo | 1474.0 MiB |

L'energia è quella dell'intera GPU durante il loop sostenuto e non sottrae il
consumo idle. Non include l'energia della CPU, anche se il profiler registra
fallback CPU. La memoria è stata isolata sul processo del notebook e comprende
tutte le sessioni e i buffer che erano residenti al momento di questo run. Il
valore di 1474 MiB non va quindi confrontato direttamente con future misure per
variante. Mancano ancora le stesse quattro metriche per ONNX FP32 e FP16; il
notebook è stato aggiornato per rimisurare anche INT8 e poi FP32/FP16 lasciando
residente una sola sessione ONNX alla volta.

### Confronto finale con sessioni GPU isolate

La successiva esecuzione della sezione di efficienza ha lasciato residente una
sola sessione ONNX alla volta. Questi valori sostituiscono il precedente picco
INT8 di 1474 MiB per i confronti tra precisioni.

| Variante | Pixel accuracy | mIoU | Media (ms) | P95 (ms) | Energia/immagine (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX FP32 CUDA | 0.932902 | 0.719813 | 11.874 | 12.068 | 878.795 | 1.137922 | 1364.0 |
| ONNX FP16 CUDA | 0.932941 | 0.720057 | 8.163 | 10.181 | 523.068 | 1.911797 | 1236.0 |
| INT8 Entropy + TRT FP16 | 0.925775 | 0.691266 | 4.170 | 5.137 | 199.959 | 5.001025 | 1168.0 |

Rispetto a ONNX FP32, FP16 riduce l'energia per immagine del 40.48% e la
memoria GPU di picco del 9.38%, senza una perdita materiale di accuratezza.
INT8 riduce l'energia per immagine del 77.25% e la memoria di picco del 14.37%,
con una perdita assoluta di 0.028547 mIoU. Rispetto a FP16, INT8 usa il 61.77%
di energia in meno per immagine e il 5.50% di memoria in meno.

La potenza media campionata è conservata nel `benchmark_results.json` prodotto
dal notebook ma non era inclusa nell'output compatto condiviso per questo
aggiornamento. PyTorch FP32 non dispone ancora delle metriche di efficienza e
non entra nel confronto energetico.

## DeepLabV3+ MobileNetV2 — Entropy, operatori estesi e fallback FP16

Run dell'11 settembre 2026 su validation, 200 immagini. La configurazione PTQ
resta QDQ S8S8 simmetrica, pesi per-channel, calibrazione Entropy su 500
immagini train random (seed 42), operatori richiesti Conv, ConvTranspose,
Resize, MaxPool e AveragePool, TensorRT INT8 con fallback FP16.

La conversione ONNX FP16 ha richiesto il workaround mirato per più `Resize` che
condividono la costante `val_679`: sono stati rimossi un Cast e un value_info
duplicati equivalenti, poi il tipo del value_info di `val_679` è stato
riallineato da FLOAT16 al tipo FLOAT dell'initializer. Il checker ONNX completo
è passato. L'artefatto FP16 contiene 131 initializer FP16, uno FP32 e tre Cast;
misura 8.679353 MiB, il 48.937378% in meno dell'ONNX FP32.

### Confronto su validation

| Variante | Pixel accuracy | mIoU | ΔmIoU vs PyTorch | Media (ms) | P95 (ms) | img/s | Energia/immagine (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 CUDA | 0.925843 | 0.808135 | +0.000000 | 9.480 | 10.810 | 105.48 | n.d. | n.d. | n.d. |
| ONNX FP32 CUDA | 0.925843 | 0.808134 | -0.000000 | 8.836 | 8.984 | 113.18 | 653.719 | 1.529709 | 706.0 |
| ONNX FP16 CUDA | 0.925839 | 0.808132 | -0.000002 | 6.030 | 8.400 | 165.83 | 377.875 | 2.646378 | 706.0 |
| INT8 Entropy + TRT FP16 | 0.889907 | 0.740750 | -0.067384 | 4.018 | 4.641 | 248.85 | 159.191 | 6.281763 | 628.0 |

FP16 è sostanzialmente lossless: conserva la mIoU entro 0.000002 rispetto a
PyTorch e riduce del 31.76% la latenza media e del 42.20% l'energia per immagine
rispetto a ONNX FP32. In questa misura non riduce il picco di memoria GPU.

INT8 riduce del 54.52% la latenza media, del 75.65% l'energia per immagine e
dell'11.05% il picco di memoria rispetto a ONNX FP32, ma perde 0.067384 mIoU.
Rispetto a FP16 consuma il 57.87% di energia in meno per immagine. Il probe
registra 63 eventi TensorRT e 62 CPU: l'esecuzione rimane mista e i valori sono
eventi del profiler, non conteggi di nodi.

La parità INT8 su un campione mostra agreement 0.996258, errore medio assoluto
dei logit 0.413902 ed errore massimo 3.346274. L'agreement elevato su un singolo
campione non contraddice la perdita aggregata di mIoU: non misura direttamente
la qualità per classe sull'intero dataset.

### Protocollo test

La configurazione del notebook mantiene la selezione su validation e offre un
gate separato `RUN_FINAL_TEST_EVALUATION`, disattivato di default. Dopo la scelta
del modello finalista, il gate calcola sull'intero test set pixel accuracy,
mIoU e IoU per classe per PyTorch FP32, ONNX FP32, ONNX FP16 e INT8. Il test non
deve essere usato per scegliere calibrazione, operatori o precisione.

### Risultati finali sul test set

Run su tutte le 800 immagini del test set, con configurazione e artefatti già
scelti sulla validation.

| Variante | Pixel accuracy | mIoU | ΔmIoU vs ONNX FP32 |
|---|---:|---:|---:|
| PyTorch FP32 CUDA | 0.905928526 | 0.640662845 | -0.000000115 |
| ONNX FP32 CUDA | 0.905928550 | 0.640662960 | +0.000000000 |
| ONNX FP16 CUDA | 0.905943046 | 0.640704617 | +0.000041657 |
| INT8 Entropy + TRT FP16 | 0.885792899 | 0.591381245 | -0.049281715 |

PyTorch e ONNX FP32 coincidono entro il rumore numerico. FP16 resta
sostanzialmente lossless anche sul test: il delta positivo di 0.000042 mIoU non
è materialmente significativo. INT8 perde 0.049282 mIoU e 0.020136 di pixel
accuracy rispetto a ONNX FP32.

| Classe | ONNX FP32 IoU | ONNX FP16 IoU | INT8 IoU | Δ INT8 vs FP32 |
|---|---:|---:|---:|---:|
| Bedrock | 0.705518 | 0.705590 | 0.668469 | -0.037049 |
| Hole | 0.096467 | 0.096465 | 0.096307 | -0.000161 |
| Ridge | 0.918872 | 0.919104 | 0.773057 | -0.145815 |
| Rock | 0.219684 | 0.219989 | 0.170095 | -0.049588 |
| Rover | 0.849929 | 0.849377 | 0.701958 | -0.147971 |
| Sand / Soil | 0.883888 | 0.883860 | 0.868045 | -0.015843 |
| Sky | 0.945039 | 0.945517 | 0.936970 | -0.008069 |
| Track | 0.505906 | 0.505736 | 0.516148 | +0.010242 |

La perdita INT8 è concentrata soprattutto su Rover e Ridge, seguite da Rock e
Bedrock; Track migliora leggermente. Hole e Rock sono già deboli nella baseline
FP32 sul test, quindi non sono un problema introdotto soltanto dalla
quantizzazione.

Il passaggio validation→test riduce la mIoU ONNX FP32 da 0.808134 a 0.640663
(-0.167471). Anche INT8 scende da 0.740750 a 0.591381 (-0.149369). Questo è un
segnale di generalizzazione/distribuzione tra split distinto dall'errore di
quantizzazione e va discusso separatamente. Dopo questo run, il test di
DeepLabV3+ è considerato osservato e non deve guidare ulteriori modifiche alla
configurazione.

## U-Net MobileNetV2 Conv-only — risultati finali sul test set

Run sulle 800 immagini del test set con la configurazione già congelata:
calibrazione Entropy su 500 immagini train random (seed 42), QDQ S8S8
simmetrico, pesi per-channel, sole Conv richieste in INT8 e fallback TensorRT
FP16. La chiave dell'artefatto è
`onnx_int8_entropy_random_conv_only_trt_fp16`.

| Variante | Pixel accuracy | mIoU | ΔmIoU vs ONNX FP32 |
|---|---:|---:|---:|
| PyTorch FP32 CUDA | 0.912211723 | 0.645717582 | +0.000000171 |
| ONNX FP32 CUDA | 0.912211680 | 0.645717410 | +0.000000000 |
| ONNX FP16 CUDA | 0.912217755 | 0.646071124 | +0.000353713 |
| INT8 Entropy Conv-only + TRT FP16 | 0.908498406 | 0.615524817 | -0.030192594 |

PyTorch e ONNX FP32 coincidono entro il rumore numerico. FP16 non introduce una
perdita misurabile; il piccolo delta positivo di 0.000354 mIoU non va
interpretato come un miglioramento sostanziale. INT8 perde 0.030193 mIoU e
0.003713 di pixel accuracy rispetto a ONNX FP32.

| Classe | ONNX FP32 IoU | ONNX FP16 IoU | INT8 IoU | Δ INT8 vs FP32 |
|---|---:|---:|---:|---:|
| Bedrock | 0.715951 | 0.715949 | 0.712641 | -0.003310 |
| Hole | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| Ridge | 0.921660 | 0.921649 | 0.900915 | -0.020745 |
| Rock | 0.215788 | 0.215741 | 0.201629 | -0.014159 |
| Rover | 0.825213 | 0.827940 | 0.673605 | -0.151608 |
| Sand / Soil | 0.893447 | 0.893463 | 0.890279 | -0.003168 |
| Sky | 0.942623 | 0.942617 | 0.919541 | -0.023082 |
| Track | 0.651056 | 0.651210 | 0.625588 | -0.025468 |

Quasi tutta la sensibilità INT8 è concentrata su Rover; le altre perdite sono
molto più contenute. Hole resta a zero già in FP32, quindi non è un fallimento
causato dalla quantizzazione.

Rispetto ai risultati validation precedenti, ONNX FP32 passa da 0.719813 a
0.645717 mIoU (-0.074096) e INT8 da 0.691266 a 0.615525 (-0.075741). Il gap tra
split è sensibilmente inferiore a quello osservato per DeepLabV3+.

### Confronto hold-out tra architetture

| Precisione | U-Net MobileNetV2 mIoU | DeepLabV3+ MobileNetV2 mIoU | Δ U-Net − DeepLabV3+ |
|---|---:|---:|---:|
| ONNX FP32 | 0.645717 | 0.640663 | +0.005054 |
| ONNX FP16 | 0.646071 | 0.640705 | +0.005367 |
| INT8 | 0.615525 | 0.591381 | +0.024144 |

Sul test U-Net è leggermente migliore in FP32/FP16 ed è più robusta alla PTQ
INT8: perde 0.030193 mIoU contro 0.049282 di DeepLabV3+. DeepLabV3+ riconosce
però Hole (IoU 0.096467 in FP32), mentre U-Net non la riconosce; la sola mIoU
non deve quindi sostituire il controllo dei requisiti per classe.

## SegFormer-B0 standard — INT8 Conv-only preliminare

Primo run su validation del checkpoint senza oversampling. La configurazione
effettivamente registrata è QDQ S8S8 simmetrica, pesi per-channel, Entropy su
500 immagini train random e chunk di calibrazione 5. Sono state richieste
soltanto le Conv; il valore chunk 5 dell'output è considerato autorevole anche
se una revisione successiva del notebook propone chunk 1 per contenere la RAM.

### Copertura e artefatto

| Proprietà | Valore |
|---|---:|
| Dimensione INT8 QDQ | 11.859787 MiB |
| Riduzione rispetto a ONNX FP32 | 22.691263% |
| Tempo di quantizzazione | 409.185 s |
| Conv QDQ | 20/20 |
| QuantizeLinear | 40 |
| DequantizeLinear | 80 |
| Initializer INT8 | 80 |
| Initializer UINT8 | 0 |
| Zero-point non nulli | 0 |

Il grafo FP32 contiene inoltre 68 MatMul, 30 LayerNormalization, 8 Softmax,
76 Add, 76 Reshape, 76 Transpose e 5 Resize. Questi operatori non sono stati
richiesti in INT8 e restano fuori dalla copertura QDQ Conv-only. La riduzione di
dimensione limitata al 22.69% è quindi coerente con una parte consistente dei
pesi, in particolare le proiezioni lineari, ancora memorizzata floating point.

### Accuratezza e prestazioni su validation

| Metrica | SegFormer INT8 Conv-only |
|---|---:|
| Pixel accuracy | 0.926202793 |
| mIoU | 0.798233691 |
| Prediction agreement con ONNX FP32, un campione | 0.998657227 |
| Errore medio assoluto dei logit | 0.072848491 |
| Errore massimo assoluto dei logit | 0.770195007 |
| Latenza media | 5.918855 ms |
| Latenza mediana | 5.336941 ms |
| P95 | 8.596463 ms |
| Throughput | 168.9516 immagini/s |
| Potenza media campionata | 68.598892 W |
| Energia per immagine | 327.986895 mJ |
| Immagini per joule | 3.048902 |
| Picco memoria GPU | 1118.0 MiB |

Il probe passa e registra 21 eventi TensorRT e 20 CPU. Sono eventi del profiler,
non nodi, e mostrano un'esecuzione mista: la presenza di TensorRT come primo
provider non implica che l'intero grafo sia eseguito da TensorRT.

### Confronto preliminare con gli altri INT8 su validation

| Modello INT8 | mIoU | Media (ms) | Energia/immagine (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|
| U-Net MobileNetV2 | 0.691266 | 4.170 | 199.959 | 5.001025 | 1168.0 |
| DeepLabV3+ MobileNetV2 | 0.740750 | 4.018 | 159.191 | 6.281763 | 628.0 |
| SegFormer-B0 Conv-only | 0.798234 | 5.919 | 327.987 | 3.048902 | 1118.0 |

SegFormer ha la migliore mIoU INT8: +0.106968 rispetto a U-Net e +0.057483
rispetto a DeepLabV3+. In cambio è più lento del 41.94% rispetto a U-Net e del
47.29% rispetto a DeepLabV3+, e usa rispettivamente il 64.03% e il 106.03% di
energia in più per immagine. La memoria è il 4.28% inferiore a U-Net ma il
78.03% superiore a DeepLabV3+.

### Confronto completo SegFormer su validation

| Variante | Pixel accuracy | mIoU | ΔmIoU vs PyTorch | Media (ms) | P95 (ms) | Energia/immagine (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 CUDA | 0.926924 | 0.802800 | +0.000000 | 12.524 | 13.487 | n.d. | n.d. | n.d. |
| ONNX FP32 CUDA | 0.926924 | 0.802800 | -0.000000 | 11.883 | 12.038 | 835.718 | 1.196576 | 1550.0 |
| ONNX FP16 CUDA | 0.926945 | 0.802884 | +0.000084 | 7.170 | 10.670 | 432.071 | 2.314434 | 1294.0 |
| INT8 Entropy Conv-only + TRT FP16 | 0.926203 | 0.798234 | -0.004567 | 5.919 | 8.596 | 327.987 | 3.048902 | 1118.0 |

FP16 è sostanzialmente lossless e, rispetto a ONNX FP32, riduce la latenza
media del 39.66%, l'energia per immagine del 48.30% e la memoria di picco del
16.52%. INT8 Conv-only perde soltanto 0.004567 mIoU e riduce rispettivamente
latenza, energia e memoria del 50.19%, 60.75% e 27.87%. Rispetto a FP16, INT8
usa il 24.09% di energia e il 13.60% di memoria in meno.

### SegFormer Conv-only sul test set

Run finale sulle 800 immagini test con la configurazione Conv-only già scelta
sulla validation.

| Variante | Pixel accuracy | mIoU | ΔmIoU vs ONNX FP32 |
|---|---:|---:|---:|
| PyTorch FP32 CUDA | 0.912743177 | 0.755519964 | +0.000000280 |
| ONNX FP32 CUDA | 0.912743177 | 0.755519684 | +0.000000000 |
| ONNX FP16 CUDA | 0.912749434 | 0.755540537 | +0.000020852 |
| INT8 Entropy Conv-only + TRT FP16 | 0.912228107 | 0.756030374 | +0.000510690 |

FP16 e INT8 sono entrambi equivalenti alla baseline sul test. Il delta INT8
positivo di 0.000511 mIoU è troppo piccolo per essere interpretato come un
miglioramento del modello; mostra però che la PTQ Conv-only non produce una
perdita generalizzata misurabile su questo split.

| Classe | ONNX FP32 IoU | ONNX FP16 IoU | INT8 IoU | Δ INT8 vs FP32 |
|---|---:|---:|---:|---:|
| Bedrock | 0.718464 | 0.718541 | 0.717465 | -0.000999 |
| Hole | 0.781055 | 0.781156 | 0.788567 | +0.007513 |
| Ridge | 0.901969 | 0.901982 | 0.899375 | -0.002594 |
| Rock | 0.217694 | 0.217538 | 0.216686 | -0.001007 |
| Rover | 0.830325 | 0.830322 | 0.833296 | +0.002972 |
| Sand / Soil | 0.894777 | 0.894780 | 0.894269 | -0.000508 |
| Sky | 0.940229 | 0.940262 | 0.938475 | -0.001754 |
| Track | 0.759645 | 0.759743 | 0.760108 | +0.000463 |

Il passaggio validation→test porta ONNX FP32 da 0.802800 a 0.755520 mIoU
(-0.047280) e INT8 da 0.798234 a 0.756030 (-0.042203). SegFormer mostra quindi
il gap tra split più contenuto dei tre modelli studiati e mantiene prestazioni
molto superiori a U-Net e DeepLabV3+ sul test.

### Prossimo esperimento SegFormer

La variante successiva mantiene checkpoint, calibrazione Entropy, 500 campioni
random, seed 42, chunk 5, S8S8 e fallback TensorRT FP16, aggiungendo in INT8
`MatMul` e `Gemm` alle `Conv`. Il grafo contiene 68 MatMul e nessun Gemm: Gemm
resta nella configurazione per robustezza tra exporter, ma la sua assenza viene
solo segnalata. LayerNormalization, Softmax, Add e Resize restano floating
point. Sono abilitate coppie Q/DQ dedicate per evitare la condivisione tra
consumer nel percorso TensorRT. Il test resta disattivato fino alla decisione
sulla validation.

#### Correzione compatibilità dei bias con TensorRT

Il primo tentativo Conv+MatMul+Gemm completava comunque l'esecuzione, ma durante
il partizionamento TensorRT emetteva ripetutamente errori su nodi come
`model.decode_head.classifier.bias_DequantizeLinear`. ONNX Runtime quantizza
di default i bias di Conv/Gemm come initializer INT32 seguiti da
`DequantizeLinear`; il parser TensorRT non accetta quel DQ INT32 come layer
esplicito isolato. Il fallback rendeva gli errori non necessariamente fatali,
ma poteva frammentare ulteriormente i sottografi e non era una configurazione
pulita da misurare.

Per il nuovo run `QuantizeBias=False`: Conv e MatMul restano candidati INT8,
mentre i bias rimangono floating point e possono essere eseguiti/fusi in FP16
dal fallback TensorRT. File e chiave esperimento includono `bias_float` per
non sovrascrivere il tentativo precedente. Prima del probe il notebook ora
controlla i tipi degli initializer consumati dai DQ e blocca il run se trova
sorgenti INT32/UINT8 o zero-point UINT8 incompatibili. Questo cambia il
trattamento dei bias rispetto al Conv-only precedente; il confronto principale
serve quindi a valutare l'intera variante di deployment, non a isolare il solo
effetto dei MatMul.

Un secondo tentativo ha mostrato che `QuantizeBias=False` da solo non è
sufficiente: pur senza DQ INT32 espliciti nel file, il TensorRT EP ricreava
durante il partizionamento nodi interni come `node_conv2d_bias_dq` e produceva
gli stessi errori del parser. La sorgente usata esclusivamente per la PTQ viene
quindi riscritta da `Conv(X, W, bias)` a `Conv(X, W) -> Add(bias)`, con il
bias rimodellato per il broadcast sul canale. Un gate su un campione reale
richiede equivalenza FP32 entro `rtol=atol=1e-5` prima di calibrare. Il report
salva numero e nomi delle Conv riscritte. Questa soluzione conserva Conv e
MatMul come candidati INT8, ma introduce un Add floating point per ogni Conv:
può quindi ridurre il guadagno di latenza/energia e va giudicata sulle metriche
misurate, oltre che sull'accuratezza. La variante usa il suffisso distinto
`conv_bias_add_float`, così non può riutilizzare accidentalmente il file
`bias_float` del tentativo precedente; anche la firma della engine cache
deriva dal nuovo hash del modello.

### SegFormer Conv+MatMul — risultati finali

Il run corretto usa il suffisso `conv_bias_add_float`. La riscrittura ha
separato il bias da tutte le 20 Conv e il gate FP32 rispetto al grafo originale
ha dato errore massimo e medio pari a zero, agreement 1.0 e `allclose=true`.
Il checker ONNX passa. Il grafo QDQ contiene esclusivamente initializer e
zero-point INT8: 0 UINT8, 0 zero-point non nulli e nessun nodo segnalato
dall'audit di compatibilità TensorRT.

| Proprietà | Valore |
|---|---:|
| Dimensione artefatto | 4.943312 MiB |
| Riduzione rispetto a ONNX FP32 | 67.776725% |
| Riduzione rispetto all'INT8 Conv-only | 58.3187% |
| Tempo di quantizzazione | 909.737 s |
| Conv QDQ | 20/20 |
| MatMul QDQ | 68/68 |
| Gemm QDQ | 0/0, assente nel grafo |
| QuantizeLinear / DequantizeLinear | 192 / 264 |
| Initializer INT8 | 326 |
| Conv con bias spostato in Add float | 20 |

Il probe registra un solo evento TensorRT e nessun evento CUDA/CPU:
`mixed_provider_execution=false`. L'evento rappresenta un sottografo fuso,
non un singolo operatore; in questo run l'intero grafo operativo è stato
racchiuso in un engine TensorRT. QDQ 20/20 e 68/68 dimostra la copertura
strutturale di Conv e MatMul, ma LayerNormalization, Softmax, Add, Resize e gli
altri operatori non richiesti restano floating point all'interno dell'engine.

#### Validation, 200 immagini

| Variante | Pixel accuracy | mIoU | Media (ms) | P95 (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX FP32 | 0.926924 | 0.802800 | 11.883 | 12.038 | 835.718 | 1.196576 | 1550 |
| ONNX FP16 | 0.926945 | 0.802884 | 7.046 | 7.974 | 432.071 | 2.314434 | 1294 |
| INT8 Conv-only | 0.926203 | 0.798234 | 5.919 | 8.596 | 327.987 | 3.048902 | 1118 |
| INT8 Conv+MatMul, bias Add float | 0.923067 | 0.795303 | 6.108 | 8.538 | 360.571 | 2.773377 | 524 |

Rispetto a ONNX FP32, la variante estesa riduce latenza del 48.60%, energia per
immagine del 56.85% e memoria di picco del 66.19%, con una perdita di 0.007497
mIoU. Il confronto decisivo è però con Conv-only: quantizzare i MatMul peggiora
la mIoU di 0.002930, la latenza del 3.20% e l'energia del 9.93%. In cambio
riduce la memoria di picco del 53.13% (1118 -> 524 MiB) e porta l'artefatto da
11.860 a 4.943 MiB. Per un obiettivo primariamente energetico Conv-only rimane
la configurazione preferibile; la variante estesa è interessante soltanto se
memoria GPU o spazio dell'artefatto sono vincoli più severi.

#### Test hold-out, 800 immagini

| Variante | Pixel accuracy | mIoU | ΔmIoU vs ONNX FP32 |
|---|---:|---:|---:|
| ONNX FP32 | 0.912743177 | 0.755519684 | +0.000000 |
| ONNX FP16 | 0.912749434 | 0.755540537 | +0.000021 |
| INT8 Conv-only | 0.912228107 | 0.756030374 | +0.000511 |
| INT8 Conv+MatMul, bias Add float | 0.910732098 | 0.751232890 | -0.004287 |

Anche il test favorisce Conv-only: la variante estesa perde 0.004797 mIoU
rispetto ad essa. I cali principali rispetto a ONNX FP32 sono Track
(-0.018140), Hole (-0.011917) e Sky (-0.010036); Rock (+0.006205) e Rover
(+0.005370) migliorano, ma non compensano il calo complessivo.

## U-Net ResNet34 — INT8 Conv-only

Run del checkpoint standard senza oversampling con calibrazione Entropy su 500
immagini train random. I bias delle Conv sono stati separati in Add floating
point, come nel percorso TensorRT corretto. Il probe passa con un singolo evento
TensorRT, nessun evento CUDA/CPU e `mixed_provider_execution=false`: il forward
misurato è racchiuso in un solo engine TensorRT. L'estratto ricevuto non include
dimensione e conteggi QDQ dell'artefatto INT8, quindi tali valori non vengono
inferiti.

### Validation, 200 immagini

| Variante | Pixel accuracy | mIoU | ΔmIoU vs PyTorch | Media (ms) | P95 (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 CUDA | 0.945256 | 0.815197 | +0.000000 | 16.227 | 16.437 | n.d. | n.d. | n.d. |
| ONNX FP32 CUDA | 0.945256 | 0.815197 | +0.000000 | 16.302 | 16.478 | 1213.080 | 0.824348 | 1128 |
| ONNX FP16 CUDA | 0.945256 | 0.815150 | -0.000046 | 9.189 | 11.361 | 659.759 | 1.515705 | 872 |
| INT8 Conv-only + bias Add float | 0.944716 | 0.817311 | +0.002114 | 5.819 | 7.973 | 304.097 | 3.288427 | 888 |

L'aumento validation di 0.002114 mIoU è entro la variabilità dovuta al rumore
numerico e non va interpretato come miglioramento del modello. La parità su un
campione dà agreement 0.998066, errore medio dei logit 0.129329 ed errore
massimo 3.173378. Rispetto a ONNX FP32, INT8 riduce la latenza del 64.31%,
l'energia per immagine del 74.93% e la memoria del 21.28%. Rispetto a FP16
riduce latenza ed energia del 36.67% e 53.91%, mentre il picco GPU è
leggermente superiore: 888 contro 872 MiB (+1.83%).

### Test hold-out, 800 immagini

| Variante | Pixel accuracy | mIoU | ΔmIoU vs ONNX FP32 |
|---|---:|---:|---:|
| PyTorch FP32 CUDA | 0.917278008 | 0.645492871 | -0.000000018 |
| ONNX FP32 CUDA | 0.917278028 | 0.645492888 | +0.000000000 |
| ONNX FP16 CUDA | 0.917279396 | 0.645494877 | +0.000001989 |
| INT8 Conv-only + bias Add float | 0.916762185 | 0.645086865 | -0.000406024 |

La PTQ è sostanzialmente lossless anche sul test. I delta per classe INT8
rispetto a ONNX FP32 sono contenuti: Bedrock -0.003498, Hole -0.000543, Ridge
-0.000344, Rock +0.002668, Rover -0.000506, Sand/Soil -0.000533, Sky
-0.001117 e Track +0.000624.

Il problema dominante è il checkpoint, non la PTQ: ONNX FP32 passa da
0.815197 mIoU in validation a 0.645493 sul test (-0.169704); INT8 passa da
0.817311 a 0.645087 (-0.172224). In particolare Hole scende da circa 0.735-0.753
in validation a circa 0.097 sul test. Questo indica un forte shift tra split o
overfitting e impedisce di usare la sola validation per confrontare la capacità
di generalizzazione dell'architettura.

### Posizionamento tra i modelli INT8

Sul test U-Net ResNet34 supera U-Net MobileNetV2 di 0.029562 mIoU e DeepLabV3+
MobileNetV2 di 0.053706, ma resta 0.110944 sotto SegFormer Conv-only. Consuma
52.08% di energia in più di U-Net MobileNetV2 e 91.03% in più di DeepLabV3+;
rispetto a SegFormer Conv-only consuma il 7.28% in meno, ma la perdita di
accuratezza è molto ampia. È quindi un eccellente caso di PTQ tecnicamente
riuscita, ma non il miglior candidato finale per accuratezza/energia.

## Prossimo esperimento — DeepLabV3 ResNet34

Il notebook passa al checkpoint standard `smp_deeplabv3_resnet34`, senza
oversampling. Il primo run mantiene il protocollo Entropy/random con 500
campioni, seed 42, chunk 5, QDQ S8S8 simmetrica e pesi per-channel. Vengono
quantizzate soltanto le Conv; pooling, Resize, Add e gli altri operatori
restano floating point con fallback TensorRT FP16. I bias Conv vengono
separati in Add floating point per evitare i DQ INT32 incompatibili già
osservati. La valutazione finale sul test resta disattivata fino alla verifica
della configurazione sulla validation.

## DeepLabV3 ResNet34 — INT8 Conv-only

Run del checkpoint standard senza oversampling con calibrazione Entropy/random
su 500 immagini e quantizzazione QDQ delle sole Conv. I bias Conv sono separati
in Add floating point e gli operatori non Conv restano floating point con
fallback TensorRT FP16. Il probe passa con un solo evento TensorRT, nessun
evento CUDA/CPU e `mixed_provider_execution=false`: il forward è racchiuso in
un singolo engine TensorRT. Questo non significa che ogni operatore sia INT8;
la precisione dei singoli operatori dipende dai Q/DQ e dal fallback interno
all'engine. L'estratto ricevuto non contiene dimensione, conteggi QDQ e audit
dell'artefatto INT8, quindi tali proprietà non vengono inferite.

### Conversione FP16

L'artefatto FP16 misura 49.840045 MiB, il 49.873383% in meno rispetto a ONNX
FP32. Il checker ONNX passa dopo aver sincronizzato il `value_info` di
`val_484`. La parità su un campione mostra agreement 0.999969, errore medio dei
logit 0.001750 ed errore massimo 0.021476. `allclose=false` con tolleranza
0.01 non indica da solo un problema: il disaccordo delle predizioni sul
campione di parità è circa lo 0.0031% dei pixel. Per confrontare direttamente
le metriche aggregate FP16 e FP32 sulla validation manca, nell'estratto
ricevuto, la riga ONNX FP32.

### Validation, 200 immagini

| Variante | Pixel accuracy | mIoU | Media (ms) | P95 (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX FP16 CUDA | 0.941246014 | 0.831325002 | 15.680 | 15.793 | n.d. | n.d. | n.d. |
| INT8 Conv-only + bias Add float | 0.941034794 | 0.830497838 | 7.193 | 9.271 | 455.336 | 2.196178 | 856 |

Rispetto a FP16, INT8 perde soltanto 0.000827 mIoU e riduce la latenza media
del 54.12%. L'accordo con ONNX FP32 su un campione è 0.998936, con errore medio
dei logit 0.091836 ed errore massimo 1.029616. L'estratto non include le
metriche energetiche FP32/FP16, quindi non è possibile calcolare un risparmio
energetico interno al modello senza recuperare il riepilogo completo; il dato
assoluto INT8 è 455.336 mJ per immagine.

### Test hold-out, 800 immagini

| Variante | Pixel accuracy | mIoU | ΔmIoU vs ONNX FP32 |
|---|---:|---:|---:|
| PyTorch FP32 CUDA | 0.916302466 | 0.737863283 | -0.000000006 |
| ONNX FP32 CUDA | 0.916302433 | 0.737863288 | +0.000000000 |
| ONNX FP16 CUDA | 0.916314292 | 0.737890075 | +0.000026786 |
| INT8 Conv-only + bias Add float | 0.916045871 | 0.735531282 | -0.002332007 |

La perdita INT8 sul test è contenuta: 0.002332 mIoU rispetto a ONNX FP32 e
0.002359 rispetto a FP16. Non è perfettamente lossless come U-Net ResNet34 o
SegFormer Conv-only, ma il calo relativo è circa lo 0.32% della mIoU FP32. I
delta per classe rispetto a ONNX FP32 sono Bedrock -0.001313, Hole -0.006206,
Ridge -0.001892, Rock -0.000950, Rover -0.004426, Sand/Soil -0.000073, Sky
-0.003292 e Track -0.000504. Non emergono collassi di classe dovuti alla PTQ;
Hole è la classe più sensibile.

Il passaggio validation→test porta FP16 da 0.831325 a 0.737890 mIoU
(-0.093435) e INT8 da 0.830498 a 0.735531 (-0.094967). Esiste quindi ancora un
gap tra split importante, ma molto più contenuto di quello osservato con
U-Net ResNet34. Il test è trattato come hold-out descrittivo e non va riusato
per scegliere calibrazione o iperparametri.

### Posizionamento tra i modelli INT8

| Modello INT8 Conv-only | Test mIoU | Media val (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|
| DeepLabV3+ MobileNetV2 | 0.591381 | 4.018 | 159.191 | 6.281763 | 628 |
| U-Net MobileNetV2 | 0.615525 | 4.170 | 199.959 | 5.001025 | 1168 |
| U-Net ResNet34 | 0.645087 | 5.819 | 304.097 | 3.288427 | 888 |
| DeepLabV3 ResNet34 | 0.735531 | 7.193 | 455.336 | 2.196178 | 856 |
| SegFormer-B0 | 0.756030 | 5.919 | 327.987 | 3.048902 | 1118 |

DeepLabV3 ResNet34 è il secondo modello per mIoU sul test: supera U-Net
ResNet34 di 0.090444, ma resta 0.020499 sotto SegFormer. Rispetto a SegFormer è
il 21.53% più lento e consuma il 38.83% di energia in più per immagine; usa
però il 23.43% di memoria GPU in meno. Per un deployment guidato soprattutto
dall'efficienza energetica, SegFormer Conv-only domina quindi questa variante
in accuratezza, latenza ed energia. DeepLabV3 ResNet34 resta interessante se
il vincolo principale è la memoria GPU e si accetta la perdita di accuratezza.
