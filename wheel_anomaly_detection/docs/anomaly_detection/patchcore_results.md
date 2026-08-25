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

Il grassetto identifica il miglior risultato Validation o Test attualmente
registrato per ciascuna metrica, senza mescolare i due split.

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
