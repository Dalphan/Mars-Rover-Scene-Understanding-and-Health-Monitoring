# Pre-bulk dataset — 200 immagini

## Stato

Completato e validato il 14 agosto 2026. I raster sono conservabili come
prima porzione del pool finale: gli split restano indici e non copiano file.

## Composizione

- 150 clean standalone;
- 25 coppie clean/hole, equivalenti a 50 immagini;
- totale: 175 unita e 200 RGB;
- 175 target-wheel mask e 25 anomaly mask;
- 400 artefatti raster canonici, 212,95 MiB.

Gli indici clean e paired sono disgiunti. Le quote marginali sono esatte per
camera, luce, usura, severita, superficie e settore; ruote e roll hanno scarto
massimo uno. I sampler dataset sono separati dai config storici dei pilot:

- `configs/blender/domain_randomization_dataset_v1.json`;
- `configs/blender/wheel_hole_anomaly_dataset_v1.json`.

## Run e tempi

Clean:
`outputs/anomaly_detection_2/prebulk_200/clean_batch/clean_150_v1`

- 150/150 campioni, validator OK;
- 2 chunk e 150 render call;
- wall pipeline: 1062,08 s, cioe 17 min 42 s;
- post-processing CPU cumulativo: 893,53 s.

Paired:
`outputs/anomaly_detection_2/prebulk_200/anomaly_batch/pairs_25_v1`

- preflight 360/360, 145,04 s;
- 25/25 coppie, validator OK;
- 1 chunk e 50 render call;
- wall pipeline: 661,84 s, cioe 11 min 2 s.

Tra timestamp di avvio clean e fine paired sono trascorsi 1783,05 s, cioe
29 min 43 s. La stima precedente di circa 25 minuti era quindi ottimistica di
circa 4 min 43 s, o 18,9%. Il collo di bottiglia osservato resta la
ricompressione/validazione PNG lato CPU, soprattutto nel clean.

## Pool parziale

Il manifest riusabile e:
`outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000/pool/prebulk_200_v1/manifest.jsonl`.

Contiene riferimenti repo-relative e checksum degli artefatti, senza copie.
Il riepilogo `prebulk.json` registra 175 unita e 200 immagini. Non sono ancora
stati creati split: il pool completo non ha ancora le 7500 unita clean e 1250
coppie richieste dal contratto finale.

## Verifiche

- sorgente `.blend` invariata, SHA-256
  `9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`;
- manifest clean e paired ordinati e completi;
- validator clean e paired superati;
- pool parziale accettato dal validatore delle unita;
- contact sheet A/C1/C2/D ispezionate: foro visibile, coppie allineate e nessun
  caso evidente di ruota sommersa dal terreno.

L'audit quantitativo successivo ha inoltre verificato tutti i 400 raster,
checksum, duplicati, distribuzioni e margini dei gate senza rilevare errori o
warning contrattuali. Il benchmark PNG ha escluso un cambio a zlib level 1:
risparmierebbe meno dello 0,4% end-to-end aumentando lo spazio di circa il
59%. Dettagli in [prebulk_quantitative_audit.md](prebulk_quantitative_audit.md).
