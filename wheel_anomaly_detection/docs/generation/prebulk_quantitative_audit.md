# Audit quantitativo pre-bulk e benchmark PNG

## Esito del pool

Audit eseguito sul pool canonico `prebulk_200_v1` e sui manifest sorgente
clean/paired. Report machine-readable:
`outputs/anomaly_detection_2/prebulk_200/audit/prebulk_200_audit.json`.

Esito: **PASS**, senza errori o warning contrattuali.

- 175 unita e 200 immagini RGB;
- 150 clean standalone e 25 coppie clean/hole;
- 400 raster canonici unici: 200 RGB, 175 target mask uniche e 25 anomaly
  mask; nel pool la target mask di una coppia e correttamente referenziata da
  entrambi i membri;
- tutti i raster sono decodificabili a `1200x900`; RGB in modo `RGB`, mask in
  modo `L` con soli valori `0/255`, non vuote e non piene;
- checksum del manifest pool e checksum di ogni artefatto validi;
- nessun file canonico mancante, extra o non referenziato;
- nessun gruppo byte-identico o pixel-identico, inclusi gli RGB;
- gli ID pool coincidono esattamente con i 150 record clean e i 25 record
  paired dei manifest sorgente;
- tutti i gate paired risultano veri.

Le distribuzioni image-weighted rispettano esattamente camera e usura:
A/C1/C2/D `80/55/55/10`, current/light/evident `40/90/70`. La luce e
`151/49` dusty/clear anziche `150/50` per l'arrotondamento indipendente dei
due run; lo scarto e una sola immagine. Ruote e roll hanno scarto massimo due
e tre immagini rispettivamente, coerente con il pre-bulk piccolo.

I margini paired peggiori restano validi: changed fraction `0,7429`, energia
locale `0,99937`, inclusione target ROI `1,0`, area anomalia `260 px` e lato
corto `14 px`. Un campione medium (`dr_100198`) ha mediana delta esattamente
sulla soglia, `10/255`: non invalida il pool, ma nel bulk va monitorata la
frequenza dei campioni senza margine fotometrico.

## Benchmark codifica PNG

Report machine-readable:
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

## Decisione

- mantenere PNG lossless, filtro corrente e zlib level 4;
- conservare integralmente le 200 immagini nel pool finale;
- prima del bulk, aggiungere al monitoraggio per chunk distribuzioni, RGB
  duplicati e frequenza dei gate fotometrici esattamente sulla soglia;
- eventuali ottimizzazioni future devono essere confrontate sul wall time
  end-to-end e sull'identita dei pixel, non soltanto sul micro-benchmark del
  codec.
