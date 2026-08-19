# Benchmark prestazioni renderer clean — 24 immagini

## Scopo

Il benchmark misura separatamente la cache geometrica e la sovrapposizione tra
render GPU e post-processing CPU sul piano clean approvato `dr_000078`--
`dr_000101`. Tutti i test usano Eevee 5.2, 64 sample, 1200x900 e lo stesso
matrix audit gia validato 24/24.

Report machine-readable:
`outputs/anomaly_detection_2/benchmark/clean_24/benchmark_report.json`.

## Risultati

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

## Verifica e decisione

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
PNG. Il pre-bulk dovra confermare throughput, memoria e stabilita su almeno
200 immagini; non e una garanzia lineare per tutte le 10.000 immagini.

## Run autorevoli del confronto

- `outputs/anomaly_detection_2/benchmark/clean_24/baseline_64_fresh`
- `outputs/anomaly_detection_2/benchmark/clean_24/cache_64`
- `outputs/anomaly_detection_2/benchmark/clean_24/pipeline_64_bounded`

Sono artefatti diagnostici e non appartengono al pool canonico del dataset.
