# Prestazioni e validazione della generazione

Questa pagina riunisce i confronti clean, paired e di codifica PNG eseguiti
prima del bulk v1. Le misure sono quelle dei run registrati: l'accorpamento
documentale del 14 ottobre 2026 non introduce nuovi benchmark.

I tempi interni, il wall time host e il tempo del solo codec hanno scope
diversi. Gli speedup non vanno sommati o estrapolati linearmente al dataset.
Raster e report benchmark nei percorsi storici sotto indicati sono stati
rimossi nelle pulizie locali; qui restano risultati, metodo e decisioni.
Le configurazioni dei benchmark e i renderer di produzione restano nel
progetto. I tre script host di analisi e confronto dei benchmark sono stati
rimossi nella pulizia autorizzata del 14 ottobre 2026.

La verifica del pool e i tempi misurati sulle 200 immagini sono riportati
nell'[assemblaggio finale](final_dataset_assembly.md#verifica-del-pre-bulk).
Le soglie del pilot e la policy fotometrica produttiva sono distinte nel
[contratto delle anomalie](wheel_hole_anomaly.md#gate-fotometrici-produttivi).

## Benchmark clean

### Scopo

Il benchmark misura separatamente la cache geometrica e la sovrapposizione tra
render GPU e post-processing CPU sul piano clean approvato `dr_000078`--
`dr_000101`. Tutti i test usano Eevee 5.2, 64 sample, 1200x900 e lo stesso
matrix audit gia validato 24/24.

Report storico:
`outputs/anomaly_detection_2/benchmark/clean_24/benchmark_report.json`.

### Risultati

| Variante | Pipeline interna | Wall time host | Risparmio interno | Output |
|---|---:|---:|---:|---|
| baseline sincrona | 182,78 s | 201,78 s | — | riferimento |
| sola cache geometrica | 183,33 s | 200,71 s | -0,30% | byte-identico |
| pipeline CPU/GPU, coda 4 | 121,08 s | 138,27 s | 33,75% | byte-identico |

La cache geometrica non migliora il renderer clean: il lavoro geometrico e
troppo piccolo rispetto a render e ricompressione PNG. Non viene quindi
abilitata nella configurazione clean produttiva.

Il percorso sincrono impiega complessivamente 67,92 secondi nei render e
110,40 secondi nel post-processing CPU. La pipeline sposta su un singolo
worker esclusivamente operazioni senza `bpy`: separazione del PNG RGBA in RGB
e target mask, compressione, validazione, hash, spostamento atomico e append
manifest. Mentre il worker processa il campione N, Blender prepara e rende il
campione N+1 sul thread principale.

La coda e limitata a quattro campioni. Quando e piena Blender applica
backpressure, evitando crescita illimitata di PNG intermedi durante i chunk da
100 immagini. `bpy`, scene state, materiali, camera, ray-cast e render non
vengono mai chiamati dal thread secondario.

### Verifica e decisione

- 24/24 RGB byte-identici alla baseline;
- 24/24 target-wheel mask byte-identiche;
- seed, sampling e matrici camera/ruota/rover identici;
- validator clean superato per tutti i run;
- sorgente `.blend` invariata;
- nessun cambiamento a sample, compressione o risoluzione.

La pipeline asincrona con coda 4 e abilitata in
`configs/blender/clean_batch.json`. La cache resta disponibile soltanto come
strumento benchmark e non viene promossa sul clean.

Il guadagno misurato, circa 34%, e specifico di questa macchina e di questi
PNG. Il successivo pre-bulk da 200 immagini e completato e documentato
nell'[assemblaggio finale](final_dataset_assembly.md#pre-bulk-riusato-nel-dataset-finale).
Il confronto non e una garanzia lineare per tutte le 10.000 immagini.

### Run autorevoli del confronto

- `outputs/anomaly_detection_2/benchmark/clean_24/baseline_64_fresh`
- `outputs/anomaly_detection_2/benchmark/clean_24/cache_64`
- `outputs/anomaly_detection_2/benchmark/clean_24/pipeline_64_bounded`

Sono artefatti diagnostici e non appartengono al pool canonico del dataset.

## Benchmark paired

### Scopo e stato

Il benchmark confronta quattro configurazioni sullo stesso piano deterministico
da 24 coppie T3. Non genera dati destinati al dataset e non modifica la
configurazione produttiva. Tutti i run sono stati validati con 24 coppie, 48
RGB, 24 target-wheel mask e 24 anomaly mask; il checksum del `.blend` sorgente
e rimasto invariato.

Report storico:
`outputs/anomaly_detection_2/benchmark/anomaly_24/benchmark_report.json`.

### Metodo

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

### Risultati

| Variante | Tempo 24 coppie | Risparmio vs baseline | Speedup | Esito immagine |
|---|---:|---:|---:|---|
| baseline, 64 sample | 1239,74 s | — | 1,00x | riferimento |
| cache geometrica, 64 sample | 623,23 s | 616,51 s / 49,73% | 1,99x | RGB e mask byte-identici |
| cache geometrica, 32 sample | 591,27 s | 648,47 s / 52,31% | 2,10x | differenza visivamente impercettibile |
| cache, 32 sample, PNG rapido | 560,14 s | 679,60 s / 54,82% | 2,21x | pixel identici alla variante 32 |

La cache a 64 sample conserva tutti i 48 RGB e tutte le 48 maschere
byte-identici. Rispetto al run canonico T3 precedente, meno rallentato della
baseline fresca, il guadagno conservativo e circa il 30%; l'intervallo
realistico ipotizzato prima del pre-bulk era quindi 30–50%.

A 32 sample la differenza RGB media e 0,0298 livelli su 255, PSNR 62,15 dB e
solo lo 0,0035% dei canali cambia di almeno 6 livelli. Le maschere differiscono
quasi esclusivamente sul bordo antialias: IoU minima 0,999886 per la ruota e
0,998418 per l'anomalia. Il guadagno aggiuntivo rispetto a cache 64 e pero solo
5,13%.

La compressione PNG rapida aggiunge un altro 5,26% rispetto a cache 32, ma
aumenta lo spazio RGB del 15,52%. I pixel decodificati restano identici.

### Decisione tecnica

`cache_64` e la candidata raccomandata per la produzione: conserva esattamente
il contratto raster approvato e rimuove lavoro geometrico statico ripetuto. La
cache contiene soltanto vertici locali immutabili delle ruote, punti esterni di
contatto e dati fissi della patch; placement, ray-cast, camera, roll e gate
restano calcolati per ogni coppia.

Le varianti a 32 sample e PNG rapido non vengono promosse automaticamente:
offrono circa il 5% ciascuna oltre la cache sicura, a fronte rispettivamente di
piccole variazioni raster e maggiore storage.

Il benchmark clean equivalente e stato completato ed e documentato in
[benchmark clean](#benchmark-clean). Sul percorso
clean la cache e risultata neutra, mentre la sovrapposizione CPU/GPU ha ridotto
il tempo interno del 33,75%. I due speedup restano specifici dei rispettivi
renderer e non vanno sommati o estrapolati linearmente alle 10.000 immagini;
il successivo gate fu il pre-bulk, ora completato.

### Artefatti locali

- `outputs/anomaly_detection_2/benchmark/anomaly_24/baseline_64`
- `outputs/anomaly_detection_2/benchmark/anomaly_24/cache_64`
- `outputs/anomaly_detection_2/benchmark/anomaly_24/cache_32`
- `outputs/anomaly_detection_2/benchmark/anomaly_24/cache_32_fastpng`

Questi artefatti sono diagnostici, ignorati da Git e non devono essere inclusi
nel pool canonico del dataset.

### Regressione dopo la pipeline clean

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

### Pipeline CPU/GPU paired

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

## Codifica PNG

Report storico:
`outputs/anomaly_detection_2/benchmark/png_postprocess_prebulk_200/benchmark_report.json`.

Il campione deterministico comprende 48 RGB, 48 target mask e tutte le 25
anomaly mask. Ogni variante e lossless e l'identita dei pixel decodificati e
stata verificata. I temporanei sono stati eliminati.

| Variante | Tempo encode | Spazio campione | Tempo vs corrente | Spazio vs corrente |
|---|---:|---:|---:|---:|
| unfiltered zlib 4 corrente | 3,25 s | 50,55 MiB | — | — |
| unfiltered zlib 1 | 1,68 s | 80,33 MiB | -48,2% | +58,9% |
| filtri Pillow, zlib 1 | 2,52 s | 76,07 MiB | -22,5% | +50,5% |
| filtri Pillow, zlib 4 | 4,24 s | 44,24 MiB | +30,5% | -12,5% |

Il benchmark isola intenzionalmente la codifica finale: decode del PNG RGBA,
split RGB/AOV, validazione, hash, gate e commit sono fuori dalla sezione
cronometrata. Questo evita di attribuire a zlib lavoro che zlib non esegue.

Estrapolando i tempi per tipo ai 400 raster canonici, il codec corrente pesa
circa 13 secondi sull'intero pre-bulk. zlib 1 ne risparmierebbe circa 6--7,
meno dello `0,4%` dei 1783 secondi end-to-end, ma aggiungerebbe circa il 59%
di spazio ai file ricodificati. Non viene quindi promosso.

Il profilo reale paired conferma il collo di bottiglia: sui 25 pair lo split
dei due PNG occupa cumulativamente `253,2 s`, le ispezioni `72,4 s` e i gate
`126,8 s`, contro circa 13 secondi di encode stimati per tutto il pre-bulk.
La prossima ottimizzazione utile deve ridurre decode/unfilter/split e
ispezioni ripetute, mantenendo la pipeline asincrona; cambiare soltanto il
livello zlib non produce un guadagno end-to-end significativo.

### Decisione

- mantenere PNG lossless, filtro corrente e zlib level 4;
- conservare integralmente le 200 immagini nel pool finale;
- prima del bulk fu richiesto di monitorare per chunk distribuzioni, RGB
  duplicati e frequenza dei gate fotometrici esattamente sulla soglia;
- eventuali ottimizzazioni future devono essere confrontate sul wall time
  end-to-end e sull'identita dei pixel, non soltanto sul micro-benchmark del
  codec.
