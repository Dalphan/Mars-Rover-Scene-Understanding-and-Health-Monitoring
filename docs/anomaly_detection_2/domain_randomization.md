# Domain randomization finale

Stato: pilot renderizzato, gate automatici superati e approvazione visiva
completata.

## Contratto di produzione

Il sampler in `src/wheel_preparation/domain_randomization.py` usa seed
deterministici e campionamento probabilistico, senza prodotto cartesiano. Le
probabilita congelate usate dal pilot sono:

- luce: 75% `mars_dusty_refined`, 25% `mars_clear_refined`;
- superficie: 35% corrente, 45% usura leggera, 20% usura evidente;
- camera: 40% A, 27,5% C1, 27,5% C2, 5% D;
- ruota target: uniforme sulle sei ruote;
- roll sano: otto fasi uniformi da 0 a 315 gradi.

Il contratto del dataset bulk e separato in
`configs/blender/dataset_composition_v1.json`: mantiene luce e camera, ma usa
20% superficie corrente, 45% usura leggera e 35% usura evidente. Il file del
pilot non viene riscritto per preservare gli hash dei run gia verificati.

Il jitter massimo e: Sole +/-8 gradi in azimuth e +/-5 gradi in elevazione,
energia +/-8%, World +/-10%, esposizione +/-0,15 EV; camera entro 2 cm,
direzione entro 1,5 gradi e focale entro +/-2%.

Tutti i parametri sono indipendenti dall'etichetta e bloccati nella coppia
controfattuale healthy/anomaly. Un retry del gate geometrico cambia soltanto
il jitter camera: non cambia ruota, posa, luce, usura, roll o seed di usura.

## Gate geometrici e patch 4x4 m

Prima del render Blender proietta sia la ruota sia i quattro angoli del
frustum sul terreno. Le pose A/C devono contenere tutta la ruota; D applica il
gate dedicato al crop intenzionale. Il frustum deve restare nella patch con
almeno 0,15 m di margine.

La patch procedurale e centrata sul lato sinistro. Per ogni campione il rover
viene traslato temporaneamente in XY affinche la ruota target coincida con
l'anchor approvato di `wheel_middle_left`. Il terreno georeferenziato, la
patch, UV e microgeometria non vengono spostati. La traslazione e salvata per
campione nei metadati. Questo evita copie o allargamenti della patch, ma
introduce una posizione assoluta del rover condizionata alla ruota: coordinate
world non devono essere usate come feature del modello.

Il preflight corrente ha accettato 450 candidati su 800. La tavola di 12
immagini usa quote esplicite per essere leggibile e non rappresenta una
frequenza empirica casuale: 5 A, 3 C1, 3 C2, 1 D; 9 dusty e 3 clear; 4/5/3
varianti di usura; due immagini per ruota.

## Riproduzione e output

```powershell
python scripts/host/run_domain_randomization_preview.py `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe'
```

Config: `configs/blender/domain_randomization.json`. Report, plan e preflight
restano in `outputs/anomaly_detection_2/domain_randomization_preview/`. Le
immagini e maschere della preview storica sono state rimosse dopo la promozione
del sampler; il comando sopra le rigenera deterministicamente.

La sorgente `wheel_roll_pose_sampling.blend` resta immutata (SHA-256
`9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`).

## Gate ancora necessari per le anomalie

Il pilot usa roll sani. Il bulk con anomalie deve continuare a usare il sampler
di visibilita anomaly-aware gia presente e deve aggiungere un gate di contrasto
fotometrico. Un'anomalia che passa il framing geometrico ma non e distinguibile
in luce/ombra deve fallire chiuso; non e ammesso un fallback al roll sano.

## Illuminazione finale e ombre

Il livello L5 e stato approvato visivamente e incorporato direttamente nei
preset definitivi. `mars_dusty_refined` usa World 0,84 e Sole 3,50;
`mars_clear_refined` usa World 0,78 e Sole 3,60. Esposizione, colori e limiti di
jitter restano invariati. Il rerender completo dei 12 campioni supera tutti i
gate, non contiene pixel sopra luminanza 239 e mantiene una mediana compresa
tra 30 e 66. I livelli sperimentali precedenti e i relativi output sono stati
rimossi dopo la selezione.
