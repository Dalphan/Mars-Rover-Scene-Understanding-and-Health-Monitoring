# S5Mars — risultati QAT ONNX/TensorRT

Ultimo aggiornamento: 19 settembre 2026.

> Le sezioni iniziali documentano i tentativi storici con `torch.ao` e con
> encoding QDQ tramite ONNX Runtime. Il percorso corrente e autorevole usa
> NVIDIA ModelOpt, export Q/DQ diretto e build TensorRT diretto; i risultati
> correnti sono riportati in fondo al documento.

## Protocollo

Il notebook autorevole è notebooks/kaggle_s5mars_onnx_qat.ipynb. La QAT
parte da un checkpoint FP32 già addestrato e non ripete la PTQ. Il primo
esperimento usa DeepLabV3+ con encoder MobileNetV2, senza oversampling e senza
augmentation.

Configurazione iniziale:

- input statico 1x3x512x512 per il deployment;
- 10 epoch massime, early stopping patience 3;
- AdamW, learning rate 1e-5, weight decay 0.01, warm-up e cosine decay;
- batch train 4 e gradient accumulation 2;
- loss supervisionata originale combined CE/Generalized Dice;
- loss totale 0.7 * supervised + 0.3 * distillation;
- distillazione KL dal teacher FP32 congelato, temperatura 2;
- BatchNorm congelate;
- fake quantization S8S8 simmetrica;
- attivazioni per-tensor e pesi Conv per-channel;
- sole Conv candidate INT8;
- bias Conv applicati come Add float;
- operatori non Conv e fallback TensorRT in FP16;
- validation per selezione del checkpoint;
- test disattivato fino al congelamento della configurazione.

## Flusso di esportazione

La QAT usa torch.ao.quantization.FakeQuantize, già incluso in PyTorch, e non
richiede ModelOpt o torchao. Il primo percorso tentava di esportare
direttamente i fake quantizer come coppie QuantizeLinear/DequantizeLinear.
Benché ONNX checker, audit QDQ e inferenza CUDA accettassero il grafo,
TensorRT 10.9 non riusciva a costruirne l'engine.

Il percorso di deployment corrente conserva i pesi aggiornati durante QAT,
li copia in normali Conv FP32 e applica poi la stessa quantize_static di ONNX
Runtime già validata nel notebook PTQ. La calibrazione usa 500 immagini train
random senza reinserimento, S8S8 simmetrico, pesi per-channel e sole Conv; i
bias sono Add float e gli operatori esclusi possono usare FP16. Il primo run
ha usato Entropy; il successivo usa MinMax, coerente con i
MovingAverageMinMaxObserver impiegati durante QAT. Il
risultato va descritto come «QAT-trained weights + ORT static QDQ encoding»:
non è un export QAT puro, perché le scale di deployment vengono ricalibrate.

## Gate di validità

Il run viene bloccato se almeno una delle seguenti condizioni non è rispettata:

- tutte le nn.Conv2d del modello sorgente sono state sostituite dal wrapper
  QAT;
- ogni Conv ONNX riceve attivazioni e pesi attraverso DQ;
- ogni peso Conv dietro DQ è materializzato come initializer INT8;
- l'export FP32 intermedio contiene i pesi del checkpoint QAT ma nessun Q/DQ;
- quantize_static produce QDQ S8S8 simmetrici per tutte le Conv;
- tutti gli zero-point Q/DQ sono INT8 e pari a zero;
- nessuna Conv conserva il bias come terzo input;
- il checker ONNX completo passa;
- la separazione Conv+bias è numericamente equivalente;
- prima di TensorRT, il QDQ eseguito con CUDA EP mantiene almeno il 95% di
  agreement delle predizioni rispetto al modello FP32 con pesi QAT;
- il profiler non registra alcuna esecuzione TensorRT.

Un solo evento nel profiler può rappresentare un intero sottografo fuso e non
prova che tutti gli operatori siano INT8. La copertura INT8 è stabilita
dall'audit Q/DQ; LayerNorm, Resize, pooling, Add e altri operatori esclusi
possono restare FP16 nello stesso engine TensorRT.

## Metriche

Il confronto usa PyTorch FP32, PyTorch QAT fake-quant e l'engine TensorRT INT8
Q/DQ con eventuale fallback FP16. Le metriche considerate sono:

- pixel accuracy, mIoU e IoU per classe;
- latenza media, mediana, P95 e immagini/s;
- average_sampled_power_w;
- energy_per_image_mj;
- images_per_joule;
- peak_gpu_memory_mib;
- dimensione dell'engine e, per INT8, copertura QDQ.

Per il confronto retrospettivo fra precisioni vengono riutilizzati i risultati
ONNX FP16 con CUDA EP già salvati nel documento PTQ. Poiché l'INT8 QAT usa
TensorRT diretto, i delta di latenza ed energia descrivono il miglioramento
della pipeline di deployment completa e non isolano matematicamente il solo
effetto del formato INT8.

Un JSON PTQ precedente può essere indicato opzionalmente per il riepilogo, ma
il notebook QAT non crea né modifica artefatti PTQ.

## Persistenza

Checkpoint QAT migliore e ultimo, history, JSON dei risultati e artefatti ONNX
vengono salvati sotto `/kaggle/working/qat_modelopt/<modello>/<variante>` e
nella cartella Drive dedicata
`s5mars_qat_modelopt_<modello>_<variante>`. Gli engine non vengono archiviati
perché dipendono da GPU e versione TensorRT; vengono rigenerati dai due ONNX.
Il benchmark TensorRT FP16 diretto resta disponibile come controllo opzionale,
ma non è richiesto per il riepilogo corrente e rimane disattivato per default.

## Risultati validation

Primo run completo, DeepLabV3+ MobileNetV2:

| Variante | Pixel accuracy | mIoU | Mean ms | Energy/image mJ | Images/J | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0.925843 | 0.808135 | 8.799 | n.d. | n.d. | n.d. |
| PyTorch QAT fake-quant | 0.926688 | 0.810535 | 31.420 | n.d. | n.d. | n.d. |
| ONNX FP32 CUDA | 0.925843 | 0.808134 | 9.227 | 620.629 | 1.611 | 762 |
| ONNX FP16 CUDA | 0.925839 | 0.808132 | 5.982 | 345.035 | 2.898 | 698 |
| ONNX QAT-trained FP32 CUDA | 0.927016 | 0.805654 | 9.774 | 617.263 | 1.620 | 762 |
| QAT weights + ORT Entropy INT8 TensorRT | 0.590848 | 0.073856 | 4.948 | 203.605 | 4.911 | 682 |

Il run Entropy INT8 è rifiutato: pur essendo eseguito da TensorRT e molto più
efficiente, predice quasi soltanto Sand / Soil. Rispetto al QAT-trained FP32,
l'agreement è 0.622406, il mean absolute error 1.823825 e il max absolute
error 14.621975. Non eseguire il test hold-out su questa configurazione.

Il secondo run con MinMax conferma il collasso solo in TensorRT:

| Variante MinMax | Pixel accuracy | mIoU | Mean ms | Energy/image mJ | Images/J | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch QAT fake-quant | 0.927771 | 0.809370 | 30.842 | n.d. | n.d. | n.d. |
| ONNX QAT-trained FP32 CUDA | 0.926948 | 0.800443 | 10.308 | 692.498 | 1.444 | 762 |
| QAT weights + ORT MinMax INT8 TensorRT | 0.590848 | 0.073856 | 4.083 | 223.560 | 4.473 | 682 |

Anche MinMax è rifiutato: agreement 0.626312, mean absolute error 1.934625
e max absolute error 13.314352. Poiché il gate QDQ CUDA della cella di export
non ha interrotto il run, l'artefatto QDQ era sopra la soglia numerica prima
di TensorRT; il degrado osservato è quindi specifico dell'esecuzione TensorRT
o di un engine memorizzato.

I messaggi «Replaced initializer ... with existing initializer» indicano
deduplicazione di costanti equivalenti. I warning
CleanUnusedInitializersAndNodeArgs sui nomi weight_bias indicano la rimozione
dei bias originali diventati inutilizzati dopo Conv+bias -> Conv + Add: non
sono la causa del collasso numerico.

## Correzioni durante il primo run

La prima esecuzione della cella di export ha prodotto un errore del checker:
una Conv consumava graph_input_cast_0 prima che il Cast produttore comparisse
nella lista dei nodi. Il collegamento era presente, ma la conversione FP16
aveva aggiunto i Cast di frontiera in fondo al grafo, violando l'ordinamento
topologico richiesto da ONNX. Prima di salvare e controllare i modelli FP16 e
QAT, il notebook ora usa ONNXModel di ONNX Runtime, lo stesso percorso già
validato nel notebook PTQ, per serializzare i produttori prima dei consumer.
La correzione non modifica pesi, scale o precisione numerica.

Un secondo arresto dell'audit ha mostrato 67/67 Conv correttamente racchiuse
da Q/DQ, ma soltanto 17 pesi già ripiegati in initializer INT8. Le altre 50
catene conservavano QuantizeLinear perché il peso costante non era collegato
direttamente al nodo: l'exporter aveva inserito nodi Identity/Cast intermedi.
Il resolver del constant folding ora attraversa queste catene costanti prima
di materializzare l'initializer INT8. L'audit rimane bloccante, ma in caso di
errore riporta conteggi e al massimo cinque esempi invece dell'intera lista.

La successiva creazione della sessione TensorRT è fallita durante il build
dell'engine, benché checker ONNX e audit QDQ fossero passati. Il checker ONNX
verifica la validità del grafo ma non il requisito più stretto del parser
TensorRT: scale e zero-point di ogni Q/DQ devono essere costanti disponibili
al build, gli zero-point devono essere tutti zero e avere la stessa shape
della scala. Il notebook ora materializza tutti questi parametri come
initializer diretti (scale FLOAT32, zero-point INT8), elimina le catene
Constant/Identity/Cast rimaste morte e ripete un audit specifico prima del
salvataggio. È inoltre attivo il detailed build log del TensorRT EP. La
normalizzazione non cambia scale apprese, pesi o collocazione dei Q/DQ e viene
inclusa nel controllo di parità contro l'export QAT non ripiegato.

Poiché anche il grafo normalizzato falliva con il solo messaggio generico
«TensorRT EP failed to create engine», il percorso di export custom è stato
ritirato dal run principale. Il notebook ora riusa la pipeline PTQ già
collaudata esclusivamente come encoder QDQ, applicandola però al modello con i
pesi del miglior checkpoint QAT. Questo richiede una nuova calibrazione delle
scale e rende l'esperimento ibrido; il JSON salva esplicitamente
strict_qat_export=false, metodo, fingerprint, seed e hash degli indici.

Il primo artefatto ibrido Entropy costruisce ed esegue correttamente l'engine,
ma fallisce sul piano numerico. Il notebook usa ora MinMax come default e
valida il QDQ con CUDA EP prima di creare la sessione TensorRT. Se l'agreement
su CUDA è inferiore a 0.95, il run si ferma dichiarando che il problema è
nell'artefatto o nella calibrazione; se CUDA passa e TensorRT degrada, il
problema è invece specifico del TensorRT EP. Cache engine e timing sono
separate per metodo di calibrazione per evitare riuso tra Entropy e MinMax.

Il probe MinMax riportava tuttavia engine_cache_hit_before_session=true e una
creazione della sessione di soli 0.226 secondi. Il gate è stato quindi
rafforzato: ora costruisce sempre un engine senza engine/timing cache, include
la soglia numerica nello status e, se INT8 con fallback FP16 fallisce, prova
automaticamente lo stesso QDQ con fallback FP32. Le cache usate dai benchmark
successivi sono isolate tramite SHA-256 del file ONNX e modalità di fallback.
La diagnosi risultante distingue:

- tensorrt_fp16_fallback_specific: QDQ corretto con fallback FP32;
- tensorrt_explicit_qdq_import_or_execution: errato anche con fallback FP32.

Il probe cache-free ha confermato la seconda diagnosi. Il nuovo engine ha
richiesto 39.400 secondi; INT8 con fallback FP16 ha prodotto agreement
0.626312, mean absolute error 1.930718 e max absolute error 13.516047. Lo
stesso QDQ con fallback FP32 ha mantenuto esattamente agreement 0.626312
(mean error 1.945011, max error 13.698138). Il problema non dipende quindi
dalla cache né dalla precisione floating-point di fallback.

Il successivo probe usa l'opzione ORT trt_op_types_to_exclude e prova
partizioni miste crescenti: Add; Resize; Add+Resize;
Add+Resize+Concat; Add+Resize+Concat+Clip. Il primo assetto con agreement
almeno 0.95 viene salvato e riutilizzato automaticamente dai benchmark.
Questa strategia replica il comportamento del precedente PTQ
DeepLabV3+ MobileNetV2, che era numericamente valido con esecuzione mista,
invece di forzare tutto il grafo in un singolo engine TensorRT.

## Percorso corrente — NVIDIA ModelOpt e TensorRT diretto

Il notebook è stato riscritto usando NVIDIA ModelOpt 0.46.1. La configurazione
QAT usa fake quantization INT8 simmetrica per Conv e shortcut residui
selezionati, distillazione dal teacher FP32, export ONNX Q/DQ diretto e un
engine TensorRT `weakly_typed_explicit_qdq` con fallback FP16. Non viene
eseguita PTQ e non viene usato `quantize_static`.

I valori di latenza ed energia descrivono la configurazione di deployment
completa; non isolano il solo effetto del QAT, perché gli esperimenti PTQ
precedenti passavano attraverso ONNX Runtime con TensorRT EP, mentre questi
esperimenti usano TensorRT direttamente.

### DeepLabV3+ MobileNetV2

Il primo run riuscito lascia in FP16 `encoder.features.14-18`, per i quali
TensorRT 10.9 non trovava tattiche INT8 valide, e mantiene INT8 54/67 Conv e
8/10 shortcut residui.

| Split/variante | mIoU | Media (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|
| Validation, TensorRT QAT | 0.800915 | 3.010 | 115.338 | 8.670 | 450 |
| Test, PyTorch FP32 | 0.640663 | n.d. | n.d. | n.d. | n.d. |
| Test, PyTorch QAT fake quant | 0.656631 | n.d. | n.d. | n.d. | n.d. |
| Test, TensorRT QAT | 0.656315 | n.d. | n.d. | n.d. | n.d. |

Sul test TensorRT QAT migliora la baseline FP32 di 0.015652 mIoU e resta a
soli -0.000316 dal modello PyTorch fake-quant, quindi l'export non introduce
un degrado rilevante.

### U-Net ResNet34

La conversione usa INT8 per tutte le Conv e per i 16 shortcut BasicBlock,
senza fallback Conv specifici.

| Variante validation | mIoU | Δ rispetto alla riga precedente |
|---|---:|---:|
| PyTorch FP32 | 0.815197 | — |
| ModelOpt fake quant iniziale | 0.814310 | -0.000887 |
| ModelOpt fake quant dopo QAT | 0.832373 | +0.018063 |
| TensorRT QAT | 0.833907 | +0.001534 |

Il confronto pre/post fake quant isola il contributo del training QAT:
`training.pre_post_qat_validation.miou_delta = +0.018063`. Il risultato
TensorRT ha prediction agreement 0.998180 rispetto al modello PyTorch QAT,
con errore medio dei logit 0.129554.

| Variante test | Pixel accuracy | mIoU |
|---|---:|---:|
| PyTorch FP32 | 0.917278 | 0.645493 |
| PyTorch ModelOpt QAT fake quant | 0.918665 | 0.651502 |
| TensorRT QAT | 0.918729 | 0.652365 |
| PTQ Conv-only precedente | 0.916762 | 0.645087 |

Sul test TensorRT QAT guadagna 0.006872 mIoU rispetto all'FP32 e 0.007278
rispetto alla PTQ. Il miglioramento per classe è concentrato soprattutto su
Track (+0.046953 rispetto all'FP32); Bedrock, Hole, Ridge, Rock, Sand/Soil e
Sky migliorano in misura minore, mentre Rover perde 0.005774.

| Deployment validation | mIoU | Media (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|
| PTQ, ORT + TensorRT EP | 0.817311 | 5.819 | 304.097 | 3.288 | 888 |
| QAT, TensorRT diretto | 0.833907 | 4.808 | 231.417 | 4.321 | 1106 |

Come configurazione di deployment complessiva, QAT riduce rispetto alla PTQ
la latenza del 17.38% e l'energia per immagine del 23.90%, aumentando le
immagini/J del 31.41%; il picco di memoria cresce però del 24.55%. Questi
delta non vanno attribuiti interamente al QAT a causa del diverso percorso
runtime.

Rispetto al QAT DeepLabV3+ MobileNetV2, U-Net ResNet34 ottiene +0.032992 mIoU
in validation ma -0.003949 sul test, è più lento del 59.74%, consuma il
100.64% di energia in più e usa il 145.78% di memoria in più. Il ribaltamento
tra validation e test conferma che U-Net ResNet34 è più sensibile allo shift
tra split e non è il candidato migliore per efficienza energetica.

## U-Net MobileNetV2

Il run usa lo stesso checkpoint dell'esperimento PTQ, dati senza oversampling
e senza augmentation e QAT con distillazione. `features.14-18` resta FP16
per compatibilità con TensorRT 10.9, mentre le altre Conv e 8/10 shortcut
InvertedResidual sono candidati INT8. L'audit bloccante del notebook è passato
e il modello è stato eseguito da un singolo engine TensorRT, senza fallback
ONNX Runtime.

### Recupero durante il QAT

| Variante validation | Pixel accuracy | mIoU | Δ rispetto alla riga precedente |
|---|---:|---:|---:|
| PyTorch FP32 | n.d. | 0.719813 | — |
| ModelOpt fake quant iniziale | 0.925232 | 0.682531 | -0.037282 |
| ModelOpt fake quant dopo QAT | 0.934414 | 0.726640 | +0.044108 |
| TensorRT QAT | 0.934203 | 0.729164 | +0.002524 |

La fake quant iniziale produce una perdita consistente, quindi MobileNetV2 è
più sensibile della ResNet34 all'inserimento dei quantizer. Il training QAT
recupera interamente la perdita e termina a +0.006827 mIoU rispetto all'FP32;
il deployment TensorRT porta il delta a +0.009351. La parità su un campione
ha prediction agreement 0.997025, errore medio dei logit 0.184602 ed errore
massimo 2.630135.

### Test hold-out

| Variante test | Pixel accuracy | mIoU | ΔmIoU vs FP32 |
|---|---:|---:|---:|
| PyTorch FP32 | 0.912212 | 0.645718 | +0.000000 |
| PyTorch ModelOpt QAT fake quant | 0.912426 | 0.652813 | +0.007095 |
| TensorRT QAT | 0.912628 | 0.649953 | +0.004235 |
| PTQ Conv-only precedente | 0.908498 | 0.615525 | -0.030193 |

TensorRT perde 0.002860 mIoU rispetto al modello fake-quant, ma conserva un
vantaggio di 0.004235 rispetto all'FP32 e di 0.034428 rispetto alla PTQ. Il
guadagno rispetto all'FP32 è dovuto soprattutto a Rock (+0.025671), Rover
(+0.008832), Track (+0.004997) e Bedrock (+0.003420); Ridge perde 0.003232 e
Sky 0.006650. Hole resta a IoU zero e il QAT non risolve quindi la mancata
predizione di questa classe.

### Deployment

| Configurazione validation | mIoU | Media (ms) | P95 (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| PTQ, ORT + TensorRT EP | 0.691266 | 4.170 | 5.137 | 199.959 | 5.001 | 1168 |
| QAT, TensorRT diretto | 0.729164 | 3.569 | 4.059 | 162.664 | 6.148 | 912 |

Come configurazione completa, QAT riduce rispetto alla PTQ la latenza del
14.41%, l'energia del 18.65% e la memoria del 21.92%, aumentando le immagini
per joule del 22.93%. Il confronto non isola il solo effetto del training,
perché cambiano sia la copertura INT8/FP16 sia il percorso runtime. L'engine
TensorRT 10.9 misura 9.923786 MiB ed è stato costruito senza cache.

### Posizionamento fra i modelli QAT

| Modello QAT TensorRT | mIoU test | Media val (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|
| DeepLabV3+ MobileNetV2 | 0.656315 | 3.010 | 115.338 | 8.670 | 450 |
| U-Net MobileNetV2 | 0.649953 | 3.569 | 162.664 | 6.148 | 912 |
| U-Net ResNet34 | 0.652365 | 4.808 | 231.417 | 4.321 | 1106 |
| SegFormer-B0 | 0.769289 | 5.657 | 291.756 | 3.428 | 1008 |
| DeepLabV3 ResNet34 | 0.736192 | 8.646 | 523.710 | 1.909 | 1130 |

SegFormer-B0 è nettamente il migliore per accuratezza sul test, con un
vantaggio di 0.033097 mIoU sul secondo classificato DeepLabV3 ResNet34;
quest'ultimo è però più lento del 52.84%, consuma il 79.50% di energia in più
e usa il 12.10% di memoria in più, quindi è dominato da SegFormer in tutte le
metriche principali.
Tra i modelli CNN, DeepLabV3+ MobileNetV2 resta il candidato più efficiente,
DeepLabV3 ResNet34 è il più accurato ma anche il meno efficiente, mentre U-Net
MobileNetV2 resta la variante U-Net più adatta al vincolo energetico.

Il gap TensorRT validation→test di U-Net MobileNetV2 è 0.079211 mIoU, vicino
al gap FP32 di 0.074095: il QAT migliora entrambi gli split, ma non elimina lo
shift di distribuzione.

## DeepLabV3 ResNet34

Il run preparato usa il checkpoint già validato nella PTQ, dati senza
oversampling o augmentation, batch train 2 con accumulo 4, distillazione e
test hold-out inizialmente disattivato. Tutte le Conv e i 16 shortcut
BasicBlock sono candidati INT8; pooling, Resize e gli altri operatori restano
FP16.

Il primo export completamente INT8 ha superato il checker ONNX e tutti gli
audit: 44/44 Conv racchiuse da Q/DQ, 44 pesi materializzati come initializer
INT8 e 16 Add residui quantizzati. La parità con il modello PyTorch QAT è
buona, con prediction agreement 0.998985 ed errore assoluto medio 0.046333.
Il risultato è tuttavia rifiutato come configurazione di deployment perché
TensorRT 10.9 non ha costruito l'engine: tutte le strategie hanno fallito
sulla Conv di proiezione
`model.encoder.layer3.0.downsample.0`, con errore «Could not find any
implementation».

Il tentativo successivo usa fallback FP16 per `model.encoder.layer3` e
`model.encoder.layer4`, cioè i due stadi profondi dilatati impiegati da
DeepLabV3, mantenendo INT8 lo stem, `layer1`, `layer2`, ASPP e segmentation
head; la variante deve essere distinta come
`int8_conv_layer3_4_fp16_residual_qdq`. Questa configurazione ha costruito un
singolo engine TensorRT ed è quella accettata per il confronto.

### Recupero durante il QAT

| Variante validation | Pixel accuracy | mIoU | Delta rispetto alla riga precedente |
|---|---:|---:|---:|
| ModelOpt fake quant iniziale | 0.941005 | 0.827119 | — |
| ModelOpt fake quant dopo QAT | 0.944630 | 0.839347 | +0.012228 |
| TensorRT QAT | 0.944529 | 0.839816 | +0.000469 |

L'engine conserva completamente il recupero del training, con prediction
agreement 0.998901 ed errore assoluto medio dei logit 0.046383 rispetto al
modello PyTorch con fallback FP16.

### Test hold-out

| Variante test | Pixel accuracy | mIoU | Delta mIoU vs FP32 |
|---|---:|---:|---:|
| PyTorch FP32 | 0.916302 | 0.737863 | +0.000000 |
| PyTorch ModelOpt QAT fake quant | 0.917142 | 0.734608 | -0.003256 |
| TensorRT QAT | 0.917128 | 0.736192 | -0.001671 |
| PTQ Conv-only precedente | 0.916046 | 0.735531 | -0.002332 |

Il miglioramento osservato in validation non si trasferisce integralmente al
test: TensorRT recupera 0.001585 mIoU rispetto al fake-quant PyTorch, ma resta
0.001671 sotto l'FP32. Track migliora sensibilmente, mentre Hole, Rover e Rock
peggiorano e compensano il guadagno delle altre classi.

### Deployment

| Configurazione validation | mIoU | Media (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|
| PTQ, ORT + TensorRT EP | 0.830498 | 7.193 | 455.336 | 2.196 | 856 |
| QAT, TensorRT diretto | 0.839816 | 8.646 | 523.710 | 1.909 | 1130 |

La QAT guadagna 0.009318 mIoU in validation e 0.000661 sul test rispetto alla
PTQ, ma questa variante con `layer3` e `layer4` in fallback è meno efficiente:
latenza +20.20%, energia +15.02%, memoria +32.01% e immagini per joule
-13.06%. Il risultato conferma la necessità di confrontare INT8 e TensorRT
FP16 nello stesso runtime prima di attribuire un vantaggio energetico alla
quantizzazione.

## SegFormer-B0 Conv-only

Il notebook è ora configurato per il checkpoint standard `segformer_b0`
senza oversampling, con batch train 2, accumulo 4, distillazione e test
hold-out disattivato. Il primo QAT quantizza soltanto i Conv2d; Linear/MatMul,
LayerNorm, Softmax, Add e Resize restano FP16 e non viene usato alcun wrapper
per shortcut CNN.

La scelta replica la configurazione PTQ SegFormer più efficiente: Conv-only
aveva mIoU validation 0.798234, 327.987 mJ per immagine e 5.919 ms, mentre
Conv+MatMul scendeva a 0.795303, 360.571 mJ e 6.108 ms. Linear/MatMul,
LayerNorm, Softmax, Add e Resize restano floating point e il deployment usa
TensorRT diretto con fallback interno FP16.

### Recupero durante il QAT

| Variante validation | Pixel accuracy | mIoU | Delta rispetto alla riga precedente |
|---|---:|---:|---:|
| ModelOpt fake quant iniziale | 0.926642 | 0.802891 | — |
| ModelOpt fake quant dopo QAT | 0.933419 | 0.831690 | +0.028799 |
| TensorRT QAT | 0.933212 | 0.830037 | -0.001654 |

Il training recupera 0.028799 mIoU rispetto alla quantizzazione iniziale e
l'engine conserva quasi tutto il guadagno, con prediction agreement 0.999340
ed errore assoluto medio dei logit 0.033701 rispetto al modello PyTorch QAT.

### Test hold-out

| Variante test | Pixel accuracy | mIoU | Delta mIoU vs FP32 |
|---|---:|---:|---:|
| PyTorch FP32 | 0.912743 | 0.755520 | +0.000000 |
| PyTorch ModelOpt QAT fake quant | 0.915709 | 0.769603 | +0.014083 |
| TensorRT QAT | 0.915656 | 0.769289 | +0.013769 |
| PTQ Conv-only precedente | 0.912228 | 0.756030 | +0.000510 |

TensorRT perde soltanto 0.000314 mIoU rispetto al fake-quant PyTorch e
migliora la PTQ di 0.013258; tutte le classi valutate migliorano rispetto
all'FP32, con il contributo più evidente di Rock, Hole, Rover e Sky.

### Deployment

| Configurazione validation | mIoU | Media (ms) | Energia/img (mJ) | Immagini/J | Picco GPU (MiB) |
|---|---:|---:|---:|---:|---:|
| PTQ, ORT + TensorRT EP | 0.798234 | 5.919 | 327.987 | 3.049 | 1118 |
| QAT, TensorRT diretto | 0.830037 | 5.657 | 291.756 | 3.428 | 1008 |

Come configurazione completa, QAT guadagna 0.031803 mIoU in validation e
0.013258 sul test rispetto alla PTQ, riducendo inoltre latenza del 4.42%,
energia per immagine dell'11.05% e memoria del 9.84%, mentre le immagini per
joule aumentano del 12.42%. Questi delta descrivono i due deployment completi
e non isolano il solo effetto del QAT, perché PTQ e QAT usano runtime diversi.

Il gap TensorRT validation-test è 0.060748 mIoU, molto più contenuto di quello
osservato nei modelli CNN QAT, benché superiore al gap FP32 di circa 0.04728.

## QAT INT8 rispetto agli FP16 già salvati

Il confronto seguente abbina l'ONNX FP16 eseguito con CUDA EP durante la PTQ
all'INT8 QAT eseguito con TensorRT diretto. Non richiede nuove esecuzioni.

| Modello | mIoU val FP16 → INT8 | Delta | mIoU test FP16 → INT8 | Delta |
|---|---:|---:|---:|---:|
| DeepLabV3+ MobileNetV2 | 0.808132 → 0.800915 | -0.007218 | 0.640705 → 0.656315 | +0.015610 |
| U-Net MobileNetV2 | 0.720057 → 0.729164 | +0.009107 | 0.646071 → 0.649953 | +0.003882 |
| U-Net ResNet34 | 0.815150 → 0.833907 | +0.018756 | 0.645495 → 0.652365 | +0.006870 |
| SegFormer-B0 | 0.802884 → 0.830037 | +0.027152 | 0.755541 → 0.769289 | +0.013748 |
| DeepLabV3 ResNet34 | 0.831325 → 0.839816 | +0.008491 | 0.737890 → 0.736192 | -0.001698 |

| Modello | Mean ms FP16 → INT8 | Delta | mJ/img FP16 → INT8 | Delta | img/J FP16 → INT8 | Picco MiB FP16 → INT8 |
|---|---:|---:|---:|---:|---:|---:|
| DeepLabV3+ MobileNetV2 | 6.030 → 3.010 | -50.09% | 377.875 → 115.338 | -69.48% | 2.646 → 8.670 (+227.62%) | 706 → 450 (-36.26%) |
| U-Net MobileNetV2 | 8.163 → 3.569 | -56.28% | 523.068 → 162.664 | -68.90% | 1.912 → 6.148 (+221.56%) | 1236 → 912 (-26.21%) |
| U-Net ResNet34 | 9.189 → 4.808 | -47.68% | 659.759 → 231.417 | -64.92% | 1.516 → 4.321 (+185.09%) | 872 → 1106 (+26.83%) |
| SegFormer-B0 | 7.170 → 5.657 | -21.10% | 432.071 → 291.756 | -32.48% | 2.314 → 3.428 (+48.09%) | 1294 → 1008 (-22.10%) |
| DeepLabV3 ResNet34 | 15.680 → 8.646 | -44.86% | n.d. → 523.710 | n.d. | n.d. → 1.909 | n.d. → 1130 |

Tra i quattro modelli con misure energetiche FP16 complete, l'INT8 QAT riduce
l'energia per immagine dal 32.48% al 69.48%. DeepLabV3+ MobileNetV2 rimane il
deployment più efficiente in valore assoluto, mentre SegFormer-B0 offre la
migliore accuratezza e un risparmio energetico più contenuto. U-Net ResNet34
è l'unico caso in cui il picco di memoria cresce, del 26.83%. Per DeepLabV3
ResNet34 il file FP16 non contiene energia e memoria, quindi da quei risultati
si può sostenere soltanto la riduzione di latenza del 44.86%.

## Prossimo esperimento SegFormer-B0: Conv + MLP + proiezioni decoder

Il notebook è predisposto per un secondo QAT incrementale denominato
`int8_conv_mlp_decoder_linear_qdq`: quantizza tutte le 20 Conv2d, i 16 Linear
`dense1/dense2` (oppure `fc1/fc2`, secondo la versione di Transformers) degli
otto blocchi MLP dell'encoder e le quattro proiezioni Linear del decoder.
Restano in FP16 i Linear di query, key, value e output dell'attention, le 30
LayerNorm, Softmax, Add e Resize, perché estendere subito INT8 all'attention
renderebbe impossibile attribuire un eventuale peggioramento al gruppo di
operatori responsabile.

Il notebook verifica sia i nomi sia i conteggi dei moduli prima del training,
accetta esclusivamente i 20 Linear previsti, materializza come initializer
INT8 anche i loro pesi e controlla nell'ONNX che Conv e Linear selezionati
siano effettivamente racchiusi da Q/DQ. Il confronto finale usa come baseline
il QAT SegFormer Conv-only già salvato e calcola automaticamente delta di
mIoU, latenza, energia, immagini per joule, memoria ed engine size.

La variante viene marcata `selected_on_validation=true` soltanto se la mIoU
non perde più di 0.005 rispetto a 0.830037 e l'energia per immagine scende
rispetto a 291.756 mJ; `RUN_FINAL_TEST` resta `False`, quindi il test hold-out
non viene consultato prima della selezione.
