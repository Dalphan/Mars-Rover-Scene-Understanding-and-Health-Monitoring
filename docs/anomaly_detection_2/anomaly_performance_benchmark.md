# Benchmark prestazioni batch paired — 24 coppie

## Scopo e stato

Il benchmark confronta quattro configurazioni sullo stesso piano deterministico
da 24 coppie T3. Non genera dati destinati al dataset e non modifica la
configurazione produttiva. Tutti i run sono stati validati con 24 coppie, 48
RGB, 24 target-wheel mask e 24 anomaly mask; il checksum del `.blend` sorgente
e rimasto invariato.

Report machine-readable:
`outputs/anomaly_detection_2/benchmark/anomaly_24/benchmark_report.json`.

## Metodo

- stesso piano, seed, placement, camera, luce, usura e geometria del foro;
- Eevee 5.2 a 1200x900;
- tempo misurato come somma di `pair_pipeline_seconds` nel manifest, per
  escludere avvio host, preflight riusato e pause tra i comandi;
- confronto decodificato pixel per pixel di RGB e maschere;
- controllo dei lock semantici e dei gate fotometrici;
- ispezione visiva delle quattro pose A, C1, C2 e D.

Il run baseline e stato interrotto dal limite temporale del comando dopo 22
coppie e completato con resume. I tempi per coppia restano validi, ma questo
rende prudente interpretare il guadagno assoluto insieme al riferimento T3
precedente, non come una promessa di throughput hardware.

## Risultati

| Variante | Tempo 24 coppie | Risparmio vs baseline | Speedup | Esito immagine |
|---|---:|---:|---:|---|
| baseline, 64 sample | 1239,74 s | — | 1,00x | riferimento |
| cache geometrica, 64 sample | 623,23 s | 616,51 s / 49,73% | 1,99x | RGB e mask byte-identici |
| cache geometrica, 32 sample | 591,27 s | 648,47 s / 52,31% | 2,10x | differenza visivamente impercettibile |
| cache, 32 sample, PNG rapido | 560,14 s | 679,60 s / 54,82% | 2,21x | pixel identici alla variante 32 |

La cache a 64 sample conserva tutti i 48 RGB e tutte le 48 maschere
byte-identici. Rispetto al run canonico T3 precedente, meno rallentato della
baseline fresca, il guadagno conservativo e circa il 30%; l'intervallo
realistico da verificare nel pre-bulk e quindi 30–50%.

A 32 sample la differenza RGB media e 0,0298 livelli su 255, PSNR 62,15 dB e
solo lo 0,0035% dei canali cambia di almeno 6 livelli. Le maschere differiscono
quasi esclusivamente sul bordo antialias: IoU minima 0,999886 per la ruota e
0,998418 per l'anomalia. Il guadagno aggiuntivo rispetto a cache 64 e pero solo
5,13%.

La compressione PNG rapida aggiunge un altro 5,26% rispetto a cache 32, ma
aumenta lo spazio RGB del 15,52%. I pixel decodificati restano identici.

## Decisione tecnica

`cache_64` e la candidata raccomandata per la produzione: conserva esattamente
il contratto raster approvato e rimuove lavoro geometrico statico ripetuto. La
cache contiene soltanto vertici locali immutabili delle ruote, punti esterni di
contatto e dati fissi della patch; placement, ray-cast, camera, roll e gate
restano calcolati per ogni coppia.

Le varianti a 32 sample e PNG rapido non vengono promosse automaticamente:
offrono circa il 5% ciascuna oltre la cache sicura, a fronte rispettivamente di
piccole variazioni raster e maggiore storage.

Il benchmark clean equivalente e stato completato ed e documentato in
[clean_performance_benchmark.md](clean_performance_benchmark.md). Sul percorso
clean la cache e risultata neutra, mentre la sovrapposizione CPU/GPU ha ridotto
il tempo interno del 33,75%. I due speedup restano specifici dei rispettivi
renderer e non vanno sommati o estrapolati linearmente alle 10.000 immagini;
il prossimo gate prestazionale e il pre-bulk.

## Artefatti locali

- `outputs/anomaly_detection_2/benchmark/anomaly_24/baseline_64`
- `outputs/anomaly_detection_2/benchmark/anomaly_24/cache_64`
- `outputs/anomaly_detection_2/benchmark/anomaly_24/cache_32`
- `outputs/anomaly_detection_2/benchmark/anomaly_24/cache_32_fastpng`

Questi artefatti sono diagnostici, ignorati da Git e non devono essere inclusi
nel pool canonico del dataset.

## Regressione dopo la pipeline clean

Il renderer paired e stato rieseguito integralmente dopo l'estrazione della
cache condivisa e l'introduzione della pipeline CPU/GPU clean:

`outputs/anomaly_detection_2/benchmark/anomaly_24/cache_64_regression_verified`.

Il run ha prodotto 24 coppie e 48 render in un solo processo Blender, senza
retry di chunk. Validator, gate e checksum sorgente sono validi. Rispetto al
precedente `cache_64`:

- tutte le 48 maschere sono byte-identiche;
- 47/48 RGB sono byte-identici;
- nel restante RGB cambiano soltanto tre canali di tre pixel, ciascuno di un
  livello su 255; nessun canale cambia di almeno 6 livelli;
- seed, descrittori, sampling e matrici sono identici;
- i margini minimi dei gate restano invariati.

Il tempo per le 24 coppie e salito da 623,23 a 670,11 secondi (+7,52%), mentre
il solo tempo dei 48 render e quasi invariato, 131,63 contro 132,36 secondi.
La variazione riguarda quindi il carico CPU/ambiente del run, non la cache o il
render Eevee, e non giustifica una modifica al contratto approvato.

Il test ha inoltre intercettato e corretto due regressioni prima del pre-bulk:
l'import della proiezione camera nel renderer paired e l'uso involontario di
`bpy.data.images` dal worker PNG clean. Il controllo PNG asincrono ora usa
soltanto parsing binario e zlib, senza API Blender fuori dal thread principale.

## Pipeline CPU/GPU paired

Il secondo benchmark ha profilato e poi sovrapposto la finalizzazione CPU di
una coppia al rendering Blender della successiva. Un solo worker CPU esegue in
ordine split PNG, ricostruzione dell'apertura, pulizia della mask, gate,
checksum e commit atomico. Il worker non usa `bpy`; proiezione, matrici e stato
Blender vengono risolti e copiati sul thread principale prima della submit.
La coda e limitata a quattro coppie e applica backpressure.

Run validato:
`outputs/anomaly_detection_2/benchmark/anomaly_24/cache_64_pipeline`.

- 24 coppie, 48 render, 48 RGB, 24 target mask e 24 anomaly mask;
- tempo host: 602,2 s contro 691,0 s del riferimento verificato;
- risparmio: 88,8 s, pari al 12,9%;
- tutte le 48 mask byte-identiche;
- 47/48 RGB byte-identici; il restante RGB differisce in tre canali di tre
  pixel, ciascuno di 1/255, senza differenze >= 6/255;
- checksum sorgente invariato:
  `9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`.

Il tentativo di scrivere RGB e mask direttamente da File Output compositor e
stato scartato: in Blender 5.2 background il nuovo item API accettava PNG ma
produceva comunque EXR multilayer. La via non viene mantenuta come codice
inattivo o fallback implicito.

Il profilo cumulativo delle 24 coppie attribuisce 242,4 s agli split PNG,
160,3 s ai gate, 74,9 s alle ispezioni, 132,2 s ai render e 114,7 s alla
risoluzione/setup geometrico. Quest'ultima non viene pre-risolta nel piano:
un prepass Blender sposterebbe semplicemente lo stesso costo prima del render,
senza ridurre il tempo end-to-end, e introdurrebbe una cache sensibile allo
stato della scena. Potra essere rivalutata soltanto se la stessa geometria
risolta verra riusata in piu run.

La cache geometrica e la pipeline asincrona sono ora abilitate esplicitamente
nella configurazione produttiva `configs/blender/anomaly_batch.json`.
