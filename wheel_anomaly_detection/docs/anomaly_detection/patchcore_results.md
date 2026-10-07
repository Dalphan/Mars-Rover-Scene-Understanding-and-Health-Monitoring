# PatchCore experiment results

Registro incrementale dei risultati PatchCore sul dataset wheel anomaly
detection. I valori sono riportati con cinque cifre decimali; i risultati
completi conservano la precisione originale negli artefatti dei singoli run.

## Metriche principali

| Modello | Max train embeddings | Memory bank | Gaussian sigma | Split | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Memoria bank |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Light | 50.000 | 2.048 | 4 | Validation | 0,56274 | 0,28416 | 0,96938 | 0,03807 | 3 MiB |
| Light | 50.000 | 2.048 | 4 | Test | 0,54870 | 0,53404 | 0,97317 | 0,07835 | 3 MiB |
| Light (512x512) | 50.000 | 5.000 eff. (cap 20.000) | 4 | Validation | 0,60116 | 0,31060 | 0,97952 | 0,06763 | 7,32 MiB |
| Light (512x512) | 50.000 | 5.000 eff. (cap 20.000) | 4 | Test | 0,57426 | 0,55070 | 0,98185 | 0,11578 | 7,32 MiB |
| Light | 1.000.000 | 50.000 | 4 | Validation | 0,83941 | 0,59377 | 0,98417 | 0,17648 | 73,24 MiB |
| Light | 1.000.000 | 50.000 | 4 | Test | 0,68368 | 0,67847 | 0,97733 | 0,20108 | 73,24 MiB |
| Light (512x512) | 1.000.000 | 20.000 | 4 | Validation | 0,78506 | 0,52021 | 0,99142 | 0,21586 | 29,30 MiB |
| Light (512x512) | 1.000.000 | 20.000 | 4 | Test | 0,70913 | 0,68681 | 0,99231 | 0,26957 | 29,30 MiB |
| Light (512x512) | 1.000.000 | 50.000 | 4 | Validation | 0,81886 | 0,55469 | 0,99454 | 0,22679 | 73,24 MiB |
| Light (512x512) | 1.000.000 | 50.000 | 4 | Test | 0,72809 | 0,70015 | 0,99454 | 0,27555 | 73,24 MiB |
| Light (512x512) | 1.000.000 | 100.000 | 4 | Validation | 0,81356 | 0,54223 | **0,99544** | 0,22596 | 146,48 MiB |
| Light (512x512) | 1.000.000 | 100.000 | 4 | Test | 0,72940 | 0,70151 | 0,99476 | 0,27888 | 146,48 MiB |
| Light | 1.000.000 | 100.000 | 4 | Validation | 0,82394 | 0,54195 | 0,99524 | 0,23047 | 146,48 MiB |
| Light | 1.000.000 | 100.000 | 4 | Test | 0,72609 | 0,69630 | **0,99495** | 0,27693 | 146,48 MiB |
| Light | 1.000.000 | 100.000 | 1 | Validation | 0,82394 | 0,54195 | 0,99434 | 0,20570 | 146,48 MiB |
| Light | 1.000.000 | 100.000 | 1 | Test | 0,72609 | 0,69630 | 0,99453 | 0,25164 | 146,48 MiB |
| Light | 1.000.000 | 100.000 | 0 | Validation | 0,82394 | 0,54195 | 0,99420 | 0,20154 | 146,48 MiB |
| Light | 1.000.000 | 100.000 | 0 | Test | 0,72609 | 0,69630 | 0,99444 | 0,24747 | 146,48 MiB |
| Reference | Non registrato | 4.096 | 4 | Validation | 0,81634 | 0,65272 | 0,98099 | 0,26803 | 16 MiB |
| Reference | Non registrato | 4.096 | 4 | Test | 0,74095 | 0,77123 | 0,97463 | 0,28265 | 16 MiB |
| Reference | Non registrato | 20.000 | 4 | Test | 0,76609 | 0,77705 | 0,98240 | 0,27449 | 78,13 MiB |
| Reference | 1.000.000 | 50.000 | 4 | Validation | 0,89730 | **0,76151** | 0,98950 | **0,31388** | 195,31 MiB |
| Reference | 1.000.000 | 50.000 | 4 | Test | **0,77942** | **0,79660** | 0,98500 | **0,30947** | 195,31 MiB |
| Reference | 1.000.000 | 100.000 | 4 | Validation | **0,89939** | 0,74660 | 0,98971 | 0,29475 | 390,63 MiB |
| Reference | 1.000.000 | 100.000 | 4 | Test | 0,77577 | 0,79222 | 0,98529 | 0,30429 | 390,63 MiB |

Il grassetto identifica il miglior risultato Validation o Test nella tabella
storica sopra, senza mescolare i due split. I run sulla sola posa A sono
riportati separatamente, perché hanno un diverso contratto di preprocessing.

## AUPRO dei run con diagnostiche estese

| Modello | Max train embeddings | Memory bank | Gaussian sigma | Split | AUPRO@0,05 | AUPRO@0,10 | AUPRO@0,30 |
|---|---:|---:|---:|---|---:|---:|---:|
| Light | 1.000.000 | 50.000 | 4 | Test | 0,59297 | 0,71222 | 0,86508 |
| Light (512x512) | 50.000 | 5.000 eff. (cap 20.000) | 4 | Test | 0,78540 | 0,86358 | 0,94185 |
| Light (512x512) | 1.000.000 | 20.000 | 4 | Test | 0,89069 | 0,93091 | 0,96978 |
| Light (512x512) | 1.000.000 | 50.000 | 4 | Test | **0,90355** | **0,94225** | **0,97621** |
| Light | 1.000.000 | 100.000 | 4 | Test | 0,87627 | 0,92664 | 0,97078 |
| Light | 1.000.000 | 100.000 | 1 | Test | 0,87777 | 0,92678 | 0,97009 |
| Light | 1.000.000 | 100.000 | 0 | Test | Non disponibile | Non disponibile | Non disponibile |
| Reference | 1.000.000 | 50.000 | 4 | Test | Non disponibile | Non disponibile | 0,90052 |
| Reference | 1.000.000 | 100.000 | 4 | Test | 0,63244 | 0,75709 | 0,89813 |

## Effetto del reservoir 50k con cap bank 20k

Con `coreset_sampling_ratio=0,1`, 50.000 candidati producono un target di
`ceil(50.000 * 0,1) = 5.000` elementi. Il limite `max_memory_bank_size=20.000`
non viene quindi raggiunto: la bank effettiva attesa è 5.000 e occupa
`7,32 MiB`, da confermare tramite `fit_summary.memory_bank_embeddings`.

Rispetto a Light 512x512 con reservoir 1M e bank 50k, sul test il run perde
`0,15382` Image AUROC, `0,14945` Image AP, `0,01269` Pixel AUROC e `0,15977`
Pixel AP. Anche AUPRO cala di `0,11815`, `0,07867` e `0,03436` ai tre cutoff.
Il risultato dimostra che questa configurazione compatta è insufficiente, ma
non separa l’effetto del reservoir da quello della bank. Per misurare una bank
20k mantenendo invariata la copertura dei candidati, usare
`max_training_embeddings=1.000.000` e `max_memory_bank_size=20.000`.

## Ablazione input size su Light 1M/50k

Il run `512x512` raggiunge sul test Image AUROC `0,72809`, Image AP `0,70015`,
Pixel AUROC `0,99454`, Pixel AP `0,27555` e AUPRO `0,90355`, `0,94225` e
`0,97621` ai tre cutoff. La bank resta pari a `73,24 MiB`.

Il vecchio run Light 384x512/50k non ha lo stesso contratto pixel: la frazione
anomala registrata è `0,00102212`, contro `0,00064382` nel run 512x512. Non è
quindi possibile attribuire direttamente l’intero incremento pixel alla sola
input size. Il confronto più informativo disponibile è con Light 384x512/100k,
che ha frazione anomala `0,00064426`. Rispetto a quest’ultimo, 512x512/50k
differisce sul test di `+0,00199` Image AUROC, `+0,00384` Image AP, `-0,00041`
Pixel AUROC e `-0,00138` Pixel AP, ma migliora AUPRO@0,05 di `0,02728`,
AUPRO@0,10 di `0,01560` e AUPRO@0,30 di `0,00543` con metà bank.

L’input quadrato altera però il formato nativo 4:3 dei render e aumenta del
33,3% pixel, attivazioni e patch di query rispetto a 384x512. Per isolare il
beneficio della risoluzione senza deformazione geometrica è preferibile un
successivo controllo 4:3, per esempio `480x640`, e una valutazione delle mappe
su una risoluzione ground-truth comune.

A parità di input 512x512 e reservoir 1M, la bank 20.000 occupa `29,30 MiB`.
Rispetto alla 50.000 perde sul test `0,01896` Image AUROC, `0,01334` Image AP,
`0,00223` Pixel AUROC e `0,00598` Pixel AP. Conserva quindi il `97,8%` del
Pixel AP usando il `40%` della memoria. AUPRO cala di `0,01286`, `0,01133` e
`0,00643` ai tre cutoff. La bank 20k è una valida configurazione compatta sulla
frontiera Pareto; la 50k resta la scelta principale orientata alla qualità.

Portare la bank da 50.000 a 100.000 raddoppia invece la memoria da `73,24 MiB`
a `146,48 MiB`. Sul test produce soltanto `+0,00132` Image AUROC, `+0,00136`
Image AP, `+0,00022` Pixel AUROC e `+0,00333` Pixel AP; in validation peggiora
Image AUROC di `0,00530`, Image AP di `0,01247` e Pixel AP di `0,00083`. In
assenza di AUPRO per il run 100k, questa configurazione non è il punto Pareto
raccomandato.

## Ablazione Gaussian sigma su Light 1M/100k

Le metriche image-level restano identiche per `sigma` 4, 1 e 0, confermando
che lo smoothing agisce esclusivamente sulla anomaly map. Sul test, `sigma=4`
supera `sigma=1` di `0,02529` Pixel AP e `sigma=0` di `0,02946`. Il vantaggio
di `sigma=1` rispetto a `sigma=0` è `0,00417` Pixel AP.

Nei risultati estesi disponibili, il vantaggio Pixel AP di `sigma=4` rispetto
a `sigma=1` compare in tutti i gruppi di severity, camera pose, lighting e
wear. `Sigma=1` ottiene AUPRO leggermente superiori a FPR `0,05` e `0,10`, ma
le differenze sono rispettivamente solo `0,00150` e `0,00014`; `sigma=4`
rimane superiore a FPR `0,30` e nelle metriche pixel globali. Il valore
raccomandato dopo questa ablazione è quindi `gaussian_sigma=4`.

## Ablazione memory bank su Reference 1M

Il passaggio da 50.000 a 100.000 elementi raddoppia la bank da `195,31 MiB`
a `390,63 MiB`. Sul test aumenta il Pixel AUROC di appena `0,00028`, ma riduce
Image AUROC di `0,00364`, Image AP di `0,00438`, Pixel AP di `0,00518` e
AUPRO@0,30 di `0,00239`. In validation aumenta Image AUROC di `0,00210` e
Pixel AUROC di `0,00021`, ma riduce Image AP di `0,01491` e Pixel AP di
`0,01914`.

Il contratto pixel coincide fra i due run: stessa frazione anomala
(`0,00102212`) e stesso numero di regioni connesse (`1.104`). Il confronto è
quindi omogeneo e non giustifica il raddoppio della bank. Per Reference, il
punto raccomandato rimane `max_training_embeddings=1.000.000` e memory bank
`50.000`.

## Light sulla sola posa A — 2026-10-06

Run `patchcore_light_resnet18_384x512_A_crop720_bank50k__seed42__20261006_185850`.
ResNet-18, layer2/layer3, input 384x512, crop `(90, 240, 720, 720)`,
embedding 384, griglia di query 48x64, `num_neighbors=9`,
reservoir 1M e bank effettiva 50k, sigma 4. Il fit su 2.800 immagini
ha richiesto **398,6082 s**; 1.433.600 embedding campionati.
Manifest SHA-256:
`afc4d44cf8330b94c52f21a314b797abdc553d517348b11489ac15e32a3c06fb`.
Validation: 296 clean + 96 hole; test: 404 clean + 404 hole.

| Split | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Wheel Pixel AUROC | Wheel Pixel AP |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0,91609 | 0,78478 | 0,99845 | 0,45681 | 0,99831 | 0,48612 |
| Test | 0,79283 | 0,77494 | 0,99783 | 0,41054 | 0,99772 | 0,45210 |

| Split | AUPRO@0,05 | AUPRO@0,10 | AUPRO@0,30 | Regioni |
|---|---:|---:|---:|---:|
| Validation | 0,95982 | 0,97780 | 0,99206 | 98 |
| Test | 0,94927 | 0,97353 | 0,99103 | 406 |

Saturazione degli score normalizzati nulla su entrambi gli split.
Frazione di pixel anomali: 0,0005322022 in validation e
0,0011199095 nel test. Pixel AP test per severity:
large 0,55265, medium 0,44375, small 0,33090.
Le distribuzioni degli score e le diagnostiche di severity, luce e wear
sono conservate a precisione completa nel JSON originale.

| Risorsa / ambito | Valore |
|---|---:|
| Parametri totali | 2.782.784 |
| Parametri / buffer | 10,6155 / 73,2594 MiB |
| Bank FP32, 50.000 x 384 | 73,2422 MiB |
| Checkpoint | 83,9034 MiB |
| Latenza predict-only media / p95 | 101,2088 / 103,6088 ms |
| Throughput predict-only | 9,8806 immagini/s |
| H2D + predict medio | 100,2939 ms |
| End-to-end medio | 140,7615 ms |
| Baseline CUDA allocata predict-only | 191,2793 MiB |
| Picco CUDA allocato / incremento predict-only | 277,5293 / 86,2500 MiB |
| Picco CUDA riservato | 712,0000 MiB |
| Energia GPU predict-only | 6,8485224215 J/image |
| Deviazione standard tra medie delle finestre | 0,0041311614 J/image |
| Potenza GPU media | 67,8032546417 W |

Tesla T4, FP32, batch 1, 10 warm-up e 50 misure per scope.
Energia via `nvml_total_energy_counter`: 5 s di warm-up,
3 finestre complete da circa 30 s, 892 inferenze e 6.108,882 J
in 90,097179 s; nessun warning.
I contatori NVML, gli aggregati ponderati e la dispersione sono stati
verificati. Energia device-wide su input residente in GPU, esclusi
CPU, loader e trasferimento, senza sottrazione dell'idle.

La bank è già inclusa nei buffer e nel picco CUDA allocato:
non si sommano nuovamente 73,2422 MiB al totale di 277,5293 MiB.
Il totale include tutti i tensori vivi nella sessione; la memoria
riservata dell'allocator è separata da quella allocata e non include
le allocazioni del driver.

Rispetto al nuovo Reference, Light riduce il picco allocato del 28,0%
e la bank del 62,5%, ma aumenta latenza predict-only del 60,9%
ed energia per immagine del 66,4%. Il nome Light descrive la backbone
e gli embedding più piccoli, non garantisce che l'intera configurazione
sia più veloce. Anche input, griglia (3.072 contro 784 query) e
`num_neighbors` (9 contro 1) differiscono. Senza profiling dedicato,
non si attribuisce l'intero aumento di costo a uno solo di questi fattori.

Nota da riprendere nella discussione dei risultati: la localizzazione dei
piccoli fori richiede una rappresentazione spaziale sufficiente nell'input.
Un'immagine più grande può conservare dettagli persi con un forte
ridimensionamento e favorire buoni risultati di localizzazione, ma
in PatchCore aumenta le query alla memory bank e può aumentare latenza
ed energia per immagine. La configurazione Light qui misurata evidenzia
questo compromesso: memoria inferiore, costo temporale ed energetico
superiore. Valutare quindi la qualità ottenuta rispetto al costo
dell'intera configurazione, senza equiparare backbone piccola a basso consumo.
L'effetto sulla qualità resta una motivazione da discutere, non una legge
generale dimostrata da questi due run: risoluzione, backbone, embedding,
scoring e geometria delle maschere cambiano insieme. Per attribuire
il guadagno alla sola risoluzione occorre un'ablazione controllata.

Il confronto pixel con Reference non è un'ablazione della sola backbone:
Light ridimensiona il crop 720 a 384x512; Reference applica resize lato
corto 256 e center crop 224. Le maschere sono valutate su griglie diverse;
frazione anomala test 0,0011199095 contro 0,00141369 e regioni
406 contro 411. I run Light storici multiposa restano separati.

[Metriche complete](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/metrics.json),
[diagnostiche originali](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/extended_diagnostics.json),
[benchmark completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/inference_benchmark_batch1.json),
[energia con contatori per finestra](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/inference_energy_batch1.json),
[console](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/user_console_output.txt),
[provenienza](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/import_provenance.json).
Le metriche incollate sono assegnate a validation/test mediante corrispondenza
esatta con I-AUROC, I-AP e P-AP dei gruppi posa A nelle diagnostiche.
Queste non contengono un run ID interno: l'associazione al run è fornita
dall'utente e verificata attraverso metriche e conteggi.
Nessun training, benchmark o ricalcolo locale eseguito durante l'import.

## Reference sulla sola posa A — 2026-10-06

Run `patchcore_reference_wide_resnet50_2_224x224_A_crop720_bank50k__seed42__20261006_122506`.
Configurazione: Wide-ResNet50-2, layer2/layer3, input 224x224, sola posa
`A_overhead`, crop `(90, 240, 720, 720)`, embedding 1.024, reservoir 1M,
bank effettiva 50k (195,31 MiB), sigma 4.
Il fit su 2.800 immagini ha richiesto 646,05 s. Validation:
296 clean + 96 hole; test: 404 clean + 404 hole. Manifest SHA-256:
`afc4d44cf8330b94c52f21a314b797abdc553d517348b11489ac15e32a3c06fb`.

| Split | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Wheel Pixel AUROC | Wheel Pixel AP |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0,96432 | 0,91836 | 0,99674 | 0,40763 | 0,99654 | 0,41202 |
| Test | 0,86428 | 0,88631 | 0,99429 | 0,42891 | 0,99402 | 0,43602 |

| Split | AUPRO@0,05 | AUPRO@0,10 | AUPRO@0,30 | Regioni |
|---|---:|---:|---:|---:|
| Validation | 0,89053 | 0,92506 | 0,96682 | 99 |
| Test | 0,82896 | 0,89356 | 0,95442 | 411 |

La saturazione degli score immagine normalizzati e 0 su entrambi gli split.
La frazione di pixel anomali e 0,00068519 in validation e
0,00141369 nel test. Sul test la Pixel AP per severity e
large 0,68933,
medium 0,40220,
small 0,19765.
Le altre diagnostiche di severity, lighting, wear e distribuzioni degli score
sono conservate senza arrotondamento nel JSON completo.

Efficienza su Tesla T4, FP32, batch 1, 10 warm-up e 50 misure per scope:
predict-only medio 65,2060 ms, p95 67,1695 ms,
15,3360 immagini/s; H2D + predict medio 59,9457 ms;
end-to-end medio 100,7893 ms (9,9217 immagini/s).
Picco CUDA allocato 385,2476 MiB,
incremento predict-only 84,2188 MiB;
checkpoint 304488415 byte. Il loader end-to-end include
I/O e trasformazioni delle annotazioni. Energia non misurata, `null`.

Fonte: metriche e benchmark forniti dall'utente in chat, piu diagnostiche
allegate; nessun nuovo training o ricalcolo eseguito durante la registrazione.
[Metriche complete](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/metrics.json),
[diagnostiche complete](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/extended_diagnostics.json),
[benchmark completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/inference_benchmark_batch1.json).

Questo run e riportato separatamente dai Reference storici: il cambio di posa,
crop e contratto pixel impedisce di attribuire la differenza alla sola bank.

## Reference posa A: nuovo fit e misura energetica — 2026-10-06

Run `patchcore_reference_wide_resnet50_2_224x224_A_crop720_bank50k__seed42__20261006_184126`.
Configurazione e manifest coincidono con il run `20261006_122506`:
Wide-ResNet50-2, layer2/layer3, input 224x224, posa A, crop 720,
embedding 1.024, reservoir 1M, bank 50k e sigma 4.
Il nuovo fit su 2.800 immagini ha richiesto **627,3508 s**.
Il riepilogo del fit coincide, ma non è stato fornito un checksum
per verificare l'identità dei tensori del checkpoint.
Le metriche di accuratezza riportate sopra appartengono al precedente
fit: l'allegato del nuovo run non include validation/test, quindi non
vengono copiate o reinterpretate come nuovi risultati.

| Metrica | Valore |
|---|---:|
| Latenza predict-only media / p95 | 62,8859 / 66,3945 ms |
| Throughput predict-only | 15,9018 immagini/s |
| H2D + predict medio | 58,3842 ms |
| End-to-end medio | 120,1384 ms |
| Picco CUDA allocato / incremento predict-only | 385,2476 / 84,2188 MiB |
| Bank FP32, 50.000 x 1.024 | 195,3125 MiB |
| Checkpoint | 290,3828 MiB |
| Energia GPU predict-only | 4,1157470629 J/image |
| Deviazione standard fra medie delle finestre | 0,0096314551 J/image |
| Potenza GPU media | 66,0991852137 W |

Protocollo: Tesla T4, FP32, batch 1, 10 warm-up e 50 misure di
latenza per scope; energia separata, 5 s di warm-up e 3 finestre da
circa 30 s tramite `nvml_total_energy_counter`.
Tutte le finestre sono complete e senza warning: 1.447 inferenze,
90,099235 s e 5.955,486 J totali. I delta dei contatori NVML e gli
aggregati sono stati verificati contro il report.
L'energia riguarda l'intera GPU su input già residente, non CPU,
loader o trasferimento, e non sottrae il consumo a riposo.
La deviazione standard è tra finestre, non tra singole immagini.
La fase energetica non effettua fit o calibrazione.

[Benchmark completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/run_20261006_184126/inference_benchmark_batch1.json),
[energia e contatori per ripetizione](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/run_20261006_184126/inference_energy_batch1.json),
[console dell'utente](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/run_20261006_184126/user_console_output.txt),
[provenienza dell'import](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/run_20261006_184126/import_provenance.json).
I report del run precedente restano invariati. Non è stata eseguita
alcuna nuova misura durante la registrazione locale.

## Contratti e note di interpretazione

- PatchCore Light usa embedding `float32` di dimensione 384: ogni elemento
  della bank occupa 1.536 byte.
- PatchCore Reference usa embedding `float32` di dimensione 1.024: ogni
  elemento della bank occupa 4.096 byte.
- `C_leading_three_quarter` rimane inclusa nelle metriche globali e nei
  sottogruppi. Le prestazioni osservate indicano che la posa è sconsigliata per
  un futuro sistema di acquisizione controllato, ma non viene esclusa dal
  benchmark.
- Fra Light 50.000 e Light 100.000 cambiano la frazione di pixel anomali
  (`0,00102212` contro `0,00064426`) e il numero di regioni connesse (`1.104`
  contro `1.046`). Prima di attribuire interamente i guadagni pixel alla bank,
  verificare hash del manifest, preprocessing delle immagini e delle maschere,
  crop/resize e `restrict_pixels_to_target_mask`.
- AUROC e AP image-level non dipendono dalla prevalenza dei pixel, ma possono
  comunque cambiare se cambia il preprocessing delle immagini.

## Aggiornamento del registro

Per ogni nuovo run annotare almeno:

- preset, backbone e dimensione embedding;
- `max_training_embeddings` e dimensione effettiva della memory bank;
- split e quattro metriche principali;
- AUPRO ai cutoff disponibili;
- hash del manifest e contratto di preprocessing, quando disponibili.
