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
