# Experiment history

Metriche: `I-AUROC` = Image AUROC, `I-AP` = Image AP, `P-AUROC` = Pixel AUROC,
`P-AP` = Pixel AP, `W-P-AUROC`/`W-P-AP` = metriche pixel limitate alla ruota.

## PatchCore

| Esperimento | Risultati validation | Risultati test |
|---|---|---|
| Light, 384x512, max train 50.000, bank 2.048, sigma 4 | I-AUROC 0,56274; I-AP 0,28416; P-AUROC 0,96938; P-AP 0,03807 | I-AUROC 0,54870; I-AP 0,53404; P-AUROC 0,97317; P-AP 0,07835 |
| Light, 512x512, max train 50.000, bank effettiva 5.000, sigma 4 | I-AUROC 0,60116; I-AP 0,31060; P-AUROC 0,97952; P-AP 0,06763 | I-AUROC 0,57426; I-AP 0,55070; P-AUROC 0,98185; P-AP 0,11578 |
| Light, 384x512, max train 1.000.000, bank 50.000, sigma 4 | I-AUROC 0,83941; I-AP 0,59377; P-AUROC 0,98417; P-AP 0,17648 | I-AUROC 0,68368; I-AP 0,67847; P-AUROC 0,97733; P-AP 0,20108 |
| Light, 512x512, max train 1.000.000, bank 20.000, sigma 4 | I-AUROC 0,78506; I-AP 0,52021; P-AUROC 0,99142; P-AP 0,21586 | I-AUROC 0,70913; I-AP 0,68681; P-AUROC 0,99231; P-AP 0,26957 |
| Light, 512x512, max train 1.000.000, bank 50.000, sigma 4 | I-AUROC 0,81886; I-AP 0,55469; P-AUROC 0,99454; P-AP 0,22679 | I-AUROC 0,72809; I-AP 0,70015; P-AUROC 0,99454; P-AP 0,27555 |
| Light, 512x512, max train 1.000.000, bank 100.000, sigma 4 | I-AUROC 0,81356; I-AP 0,54223; P-AUROC 0,99544; P-AP 0,22596 | I-AUROC 0,72940; I-AP 0,70151; P-AUROC 0,99476; P-AP 0,27888 |
| Light, 384x512, max train 1.000.000, bank 100.000, sigma 4 | I-AUROC 0,82394; I-AP 0,54195; P-AUROC 0,99524; P-AP 0,23047 | I-AUROC 0,72609; I-AP 0,69630; P-AUROC 0,99495; P-AP 0,27693 |
| Light, 384x512, max train 1.000.000, bank 100.000, sigma 1 | I-AUROC 0,82394; I-AP 0,54195; P-AUROC 0,99434; P-AP 0,20570 | I-AUROC 0,72609; I-AP 0,69630; P-AUROC 0,99453; P-AP 0,25164 |
| Light, 384x512, max train 1.000.000, bank 100.000, sigma 0 | I-AUROC 0,82394; I-AP 0,54195; P-AUROC 0,99420; P-AP 0,20154 | I-AUROC 0,72609; I-AP 0,69630; P-AUROC 0,99444; P-AP 0,24747 |
| Reference, bank 4.096, sigma 4 | I-AUROC 0,81634; I-AP 0,65272; P-AUROC 0,98099; P-AP 0,26803 | I-AUROC 0,74095; I-AP 0,77123; P-AUROC 0,97463; P-AP 0,28265 |
| Reference, bank 20.000, sigma 4 | Non riportati | I-AUROC 0,76609; I-AP 0,77705; P-AUROC 0,98240; P-AP 0,27449 |
| Reference, max train 1.000.000, bank 50.000, sigma 4 | I-AUROC 0,89730; I-AP 0,76151; P-AUROC 0,98950; P-AP 0,31388 | I-AUROC 0,77942; I-AP 0,79660; P-AUROC 0,98500; P-AP 0,30947 |
| Reference, max train 1.000.000, bank 100.000, sigma 4 | I-AUROC 0,89939; I-AP 0,74660; P-AUROC 0,98971; P-AP 0,29475 | I-AUROC 0,77577; I-AP 0,79222; P-AUROC 0,98529; P-AP 0,30429 |

## EfficientAD-S

| Esperimento | Risultati validation | Risultati test |
|---|---|---|
| Baseline, 70.000 step, batch 1, tutte le pose, metriche raw corrette | I-AUROC 0,60653; I-AP 0,34143; P-AUROC 0,93776; P-AP 0,02183; W-P-AUROC 0,90952; W-P-AP 0,03076 | I-AUROC 0,57151; I-AP 0,55719; P-AUROC 0,93876; P-AP 0,03806; W-P-AUROC 0,91126; W-P-AP 0,05457 |
| Crop fissi 900x900 senza padding, training/early stopping aggiornati, tutte le pose | I-AUROC 0,58572; I-AP 0,31151; P-AUROC 0,92585; P-AP 0,01456; W-P-AUROC 0,90475; W-P-AP 0,02261 | I-AUROC 0,55297; I-AP 0,54004; P-AUROC 0,92582; P-AP 0,02779; W-P-AUROC 0,90442; W-P-AP 0,04330 |
| Sola posa `A_overhead`, crop 900x900, 70.000 step | I-AUROC 0,79054; I-AP 0,56053; P-AUROC 0,97899; P-AP 0,08542; W-P-AUROC 0,97211; W-P-AP 0,10104 | I-AUROC 0,63637; I-AP 0,63059; P-AUROC 0,97296; P-AP 0,07468; W-P-AUROC 0,96254; W-P-AP 0,08594 |
| Sola posa `A_overhead`, crop 720x720, checkpoint selezionato a 67.000 step | I-AUROC 0,79804; I-AP 0,57910; P-AUROC 0,97112; P-AP 0,12533; W-P-AUROC 0,96967; W-P-AP 0,13720 | I-AUROC 0,64044; I-AP 0,63633; P-AUROC 0,95785; P-AP 0,06063; W-P-AUROC 0,95445; W-P-AP 0,06797 |
| Sola posa `A_overhead`, crop 720x720, calibrazione spaziale, 70.000 step; score selezionato `spatial_topk_0_0005` | I-AUROC 0,85001; I-AP 0,58619; P-AUROC 0,94518; P-AP 0,05438; W-P-AUROC 0,93999; W-P-AP 0,06059 | I-AUROC 0,61373; I-AP 0,57583; P-AUROC 0,88780; P-AP 0,02904; W-P-AUROC 0,87668; W-P-AP 0,03189 |
| Sola posa `A_overhead`, crop 720x720, score `global_max`, training fisso 70.000 step | I-AUROC 0,81232; I-AP 0,61366; P-AUROC 0,97434; P-AP 0,15210; W-P-AUROC 0,97266; W-P-AP 0,16459 | I-AUROC 0,66502; I-AP 0,64737; P-AUROC 0,96365; P-AP 0,09004; W-P-AUROC 0,96041; W-P-AP 0,10296 |

Ablazione dello score sul checkpoint con calibrazione spaziale, selezionata su
validation I-AUROC: `global_max` 0,80215; `spatial_max` 0,84977;
`spatial_topk_0_0005` 0,85001; `spatial_topk_0_001` 0,84931;
`spatial_topk_0_0025` 0,84832; `spatial_topk_0_005` 0,84635. Sul test la
configurazione global a durata fissa ottiene i migliori risultati EfficientAD-S.

## TinyGLASS

| Esperimento | Risultati validation | Risultati test |
|---|---|---|
| Primo run, 256x256, 640 epoche a durata fissa, checkpoint finale | I-AUROC 0,56745; I-AP 0,30325; P-AUROC 0,76500; P-AP 0,00224; W-P-AUROC 0,62737; W-P-AP 0,00280 | I-AUROC 0,53603; I-AP 0,53393; P-AUROC 0,79752; P-AP 0,00453; W-P-AUROC 0,67595; W-P-AP 0,00572 |
| Backbone frozen, pose crop attivo, best checkpoint su validation I-AUROC; best epoca 77, stop epoca 97 | I-AUROC 0,60557; I-AP 0,33002; P-AUROC 0,74145; P-AP 0,00818; W-P-AUROC 0,66582; W-P-AP 0,01000 | I-AUROC 0,53761; I-AP 0,53245; P-AUROC 0,75441; P-AP 0,00847; W-P-AUROC 0,68128; W-P-AP 0,01144 |
| Ablazione aggregazione score sul checkpoint frozen: max, top-k mean 3/5/10, quantile 0,95/0,99 | Migliore `max`: I-AUROC 0,60557; I-AP 0,33002. Tutte le alternative peggiori | Test riportato solo per `max`: I-AUROC 0,53761; I-AP 0,53245 |
| Fine-tuning completo backbone, 20 epoche, stesso seed e crop | I-AUROC 0,54029; I-AP 0,28665; P-AUROC 0,76888; P-AP 0,00428; W-P-AUROC 0,75106; W-P-AP 0,00517 | I-AUROC 0,53341; I-AP 0,53127; P-AUROC 0,78097; P-AP 0,01111; W-P-AUROC 0,76296; W-P-AP 0,01458 |
| Sola posa `A_overhead`, crop attivo 900x900, LAS limitata alla target wheel, backbone frozen, GAS joint corretto | I-AUROC 0,68314; I-AP 0,42363; P-AUROC 0,75179; P-AP 0,01617; W-P-AUROC 0,72524; W-P-AP 0,02135 | I-AUROC 0,56327; I-AP 0,57585; P-AUROC 0,75311; P-AP 0,01410; W-P-AUROC 0,71810; W-P-AP 0,01607 |
| Sola posa `A_overhead`, crop 900x900, LAS target wheel, backbone frozen, `hypersphere_projection: false`, sigma 4 | I-AUROC 0,75396; I-AP 0,52245; P-AUROC 0,94373; P-AP 0,06386; W-P-AUROC 0,91882; W-P-AP 0,06611 | I-AUROC 0,63811; I-AP 0,65100; P-AUROC 0,94251; P-AP 0,07318; W-P-AUROC 0,91559; W-P-AP 0,07673 |
| Stessa configurazione posa A e `hypersphere_projection: false`, ablazione sigma 1 sul vecchio notebook | I-AUROC 0,75148; I-AP 0,51210; P-AUROC 0,94434; P-AP 0,05380; W-P-AUROC 0,91827; W-P-AP 0,05839 | I-AUROC 0,62790; I-AP 0,64263; P-AUROC 0,93910; P-AP 0,07776; W-P-AUROC 0,90967; W-P-AP 0,08166 |
| Posa A, crop 640x640, griglia layer2 32x32, LAS target wheel, backbone frozen, `hypersphere_projection: false`, sigma 2 | I-AUROC 0,75943; I-AP 0,56399; P-AUROC 0,94237; P-AP 0,09738; W-P-AUROC 0,94077; W-P-AP 0,09986 | I-AUROC 0,65056; I-AP 0,65008; P-AUROC 0,93167; P-AP 0,07821; W-P-AUROC 0,93014; W-P-AP 0,08475 |
| Posa A, crop 720x720, griglia layer2 32x32, LAS target wheel, backbone frozen, `hypersphere_projection: false`, sigma 4 | I-AUROC 0,76767; I-AP 0,53876; P-AUROC 0,94790; P-AP 0,08382; W-P-AUROC 0,94413; W-P-AP 0,08878 | I-AUROC 0,66776; I-AP 0,66793; P-AUROC 0,94598; P-AP 0,06974; W-P-AUROC 0,94248; W-P-AP 0,08184 |
| Posa A, crop 720x720, griglia layer2 32x32, LAS target wheel, backbone frozen, `hypersphere_projection: false`, sigma 0; run distinto dal sigma 4 | I-AUROC 0,76174; I-AP 0,52353; P-AUROC 0,94539; P-AP 0,07336; W-P-AUROC 0,94094; W-P-AP 0,07812 | I-AUROC 0,66668; I-AP 0,66438; P-AUROC 0,94900; P-AP 0,07839; W-P-AUROC 0,94497; W-P-AP 0,08774 |
| Posa A, crop 720x720, griglia layer2 32x32, LAS `hole`, backbone frozen, `hypersphere_projection: false`, sigma 4, LR config 1e-4, patience 20; best epoca 79, stop epoca 99 | I-AUROC 0,80602; I-AP 0,59505; P-AUROC 0,95091; P-AP 0,09012; W-P-AUROC 0,94700; W-P-AP 0,09524 | I-AUROC 0,69715; I-AP 0,72054; P-AUROC 0,94849; P-AP 0,11455; W-P-AUROC 0,94431; W-P-AP 0,12890 |
| Posa A, crop 720x720, griglia layer2 32x32, LAS `hole`, backbone frozen, `hypersphere_projection: false`, sigma 4, LR config 5e-5, patience 40; best epoca 175, stop epoca 215 | I-AUROC 0,83349; I-AP 0,65251; P-AUROC 0,95318; P-AP 0,14400; W-P-AUROC 0,95037; W-P-AP 0,14780 | I-AUROC 0,72040; I-AP 0,74722; P-AUROC 0,94802; P-AP 0,14711; W-P-AUROC 0,94485; W-P-AP 0,15485 |
| Posa A, crop 720x720, griglia layer2 32x32, LAS `hole`, backbone frozen, `hypersphere_projection: false`, sigma 4, LR config 5e-5, patience 40, pesi small/medium/large `70/25/5` | I-AUROC 0,83740; I-AP 0,67355; P-AUROC 0,95418; P-AP 0,15176; W-P-AUROC 0,95182; W-P-AP 0,15579 | I-AUROC 0,72551; I-AP 0,75216; P-AUROC 0,95092; P-AP 0,14666; W-P-AUROC 0,94829; W-P-AP 0,15488 |
| Stessa configurazione `70/25/5`, griglia riportata a layer3 16x16, sigma 4 | I-AUROC 0,85762; I-AP 0,69961; P-AUROC 0,95690; P-AP 0,14764; W-P-AUROC 0,95268; W-P-AP 0,15040 | I-AUROC 0,72482; I-AP 0,76120; P-AUROC 0,94560; P-AP 0,15775; W-P-AUROC 0,94038; W-P-AP 0,16184 |
| Ablazione post-hoc sigma `0/1/2/4`, checkpoint layer2 32x32 `70/25/5`; sigma 1 selezionato su validation | Sigma 1: P-AUROC 0,95554; P-AP 0,16525; W-P-AUROC 0,95331; W-P-AP 0,16990; AUPRO 0,82131 | Sigma 1: I-AUROC 0,72551; I-AP 0,75216; P-AUROC 0,95187; P-AP 0,15720; W-P-AUROC 0,94936; W-P-AP 0,16613 |
| Ablazione post-hoc sigma `0/1/2/4`, checkpoint layer3 16x16 `70/25/5` | Sigma 0 migliore P-AUROC/AUPRO: 0,95777/0,80575; sigma 1 migliore W-P-AP: 0,15379; sigma 2 migliore P-AP: 0,15059; sigma 4 peggiore | Non valutata: selezione validation-only |
| Sola posa `D_upper_detail`, input 256x256, layer2, sigma 1, stessa configurazione LAS hole `70/25/5` | I-AUROC 0,87855; I-AP 0,80932; P-AUROC 0,97101; P-AP 0,27521; W-P-AUROC 0,97135; W-P-AP 0,27848; AUPRO 0,90707 | I-AUROC 0,76059; I-AP 0,78112; P-AUROC 0,95694; P-AP 0,21239; W-P-AUROC 0,95819; W-P-AP 0,21632; AUPRO 0,81477 |
| Posa A, crop 720, input 384x384, layer2 48x48, sigma 1, LAS hole `70/25/5`; resto invariato rispetto al best 256 | I-AUROC 0,88355; I-AP 0,73104; P-AUROC 0,98526; P-AP 0,26045; W-P-AUROC 0,98521; W-P-AP 0,29590; AUPRO 0,93294 | I-AUROC 0,77523; I-AP 0,79281; P-AUROC 0,97213; P-AP 0,25476; W-P-AUROC 0,97186; W-P-AP 0,27647; AUPRO 0,90240 |
| Posa A, crop 720, input 384x384, layer3 24x24, sigma 1, LAS hole `70/25/5` | I-AUROC 0,87577; I-AP 0,74945; P-AUROC 0,98424; P-AP 0,25978; W-P-AUROC 0,98294; W-P-AP 0,27044; AUPRO 0,92941 | I-AUROC 0,76910; I-AP 0,80042; P-AUROC 0,97136; P-AP 0,24176; W-P-AUROC 0,96885; W-P-AP 0,25245; AUPRO 0,89668 |
| Ablazione post-hoc sigma `0/1/2/4`, checkpoint 384 layer3 24x24 | Sigma 4 migliore: P-AUROC 0,98525; P-AP 0,26479; W-P-AUROC 0,98398; W-P-AP 0,27521; AUPRO 0,93025 | Non valutata: selezione validation-only |
| Posa A, crop 720, input 512x512, layer2 64x64, sigma 1, LAS hole `70/25/5` | I-AUROC 0,83163; I-AP 0,66362; P-AUROC 0,97306; P-AP 0,24083; W-P-AUROC 0,97021; W-P-AP 0,25311; AUPRO 0,91134 | I-AUROC 0,73806; I-AP 0,76923; P-AUROC 0,96689; P-AP 0,23548; W-P-AUROC 0,96342; W-P-AP 0,24577; AUPRO 0,89266 |
| Ablazione post-hoc sigma `0/1/2/4`, checkpoint 512 layer2 64x64 | Sigma 4 migliore: P-AUROC 0,97593; P-AP 0,24562; W-P-AUROC 0,97328; W-P-AP 0,25810; AUPRO 0,91683 | Non valutata: selezione validation-only |
| Posa A, crop 720, input 512x512, layer3 32x32, sigma 1, LAS hole `70/25/5` | I-AUROC 0,85568; I-AP 0,73298; P-AUROC 0,98046; P-AP 0,28256; W-P-AUROC 0,97795; W-P-AP 0,29076; AUPRO 0,92232 | I-AUROC 0,77330; I-AP 0,81126; P-AUROC 0,97793; P-AP 0,27005; W-P-AUROC 0,97503; W-P-AP 0,27756; AUPRO 0,91764 |
| Ablazione post-hoc sigma `0/1/2/4`, checkpoint 512 layer3 32x32; sigma 4 selezionato su validation | Sigma 4: P-AUROC 0,98192; P-AP 0,28841; W-P-AUROC 0,97956; W-P-AP 0,29615; AUPRO 0,92675 | Sigma 4: I-AUROC 0,77330; I-AP 0,81126; P-AUROC 0,97924; P-AP 0,27592; W-P-AUROC 0,97646; W-P-AP 0,28299; AUPRO 0,92012 |

Nel confronto diretto col run sigma 4, sigma 1 aumenta sul test P-AP di
0,00459 e W-P-AP di 0,00493, ma riduce I-AUROC di 0,01021, I-AP di 0,00837,
P-AUROC di 0,00341 e W-P-AUROC di 0,00592. AUPRO test resta sostanzialmente
invariato (0,77051 contro 0,76972). Sigma 1 rappresenta quindi un trade-off di precisione
pixel, non un miglioramento complessivo.

Il run high-resolution con crop 640 e sigma 2 migliora rispetto al precedente
crop 900/sigma 4 la I-AUROC test di 0,01244, la W-P-AUROC di 0,01455 e la
W-P-AP di 0,00802. La P-AUROC test scende di 0,01084. AUPRO test vale 0,74834:
la localizzazione resta utile, ma il crop stretto modifica la scala dei difetti
e la prevalenza pixel, quindi P-AP non va interpretata isolatamente.

Il run crop 720/sigma 4 raggiunge la migliore I-AUROC TinyGLASS finora
(0,66776 test) e AUPRO test 0,80662. Rispetto al crop 640/sigma 2 migliora
I-AUROC di 0,01720, P-AUROC di 0,01432 e W-P-AUROC di 0,01235, mentre P-AP
scende di 0,00847. Crop e sigma cambiano insieme, quindi questo confronto non
permette di attribuire causalmente il risultato a uno dei due fattori.

Il run sigma 0 migliora rispetto al run sigma 4 P-AUROC test di 0,00302,
P-AP di 0,00864, W-P-AUROC di 0,00249 e W-P-AP di 0,00590; I-AUROC resta
quasi invariata (-0,00108). Poiche cambiano anche le metriche image-level,
questo non e un confronto post-hoc sullo stesso checkpoint e include varianza
di training.

Con LAS `hole`, l'ablazione post-hoc delle aggregazioni seleziona ancora
`max` sulla validation (I-AUROC 0,80602; I-AP 0,59505). `topk_mean_3`
e quasi equivalente in I-AUROC (0,80546) ma inferiore in I-AP (0,58910);
top-k piu ampi e quantili peggiorano ulteriormente. Il test resta quindi
valutato una sola volta con `max`.

Il run LAS `hole` a LR ridotto migliora rispetto al precedente hole sul test
I-AUROC di 0,02325, I-AP di 0,02667, P-AP di 0,03256 e W-P-AP di 0,02595.
P-AUROC scende di 0,00047, mentre W-P-AUROC cresce di 0,00054. AUPRO test
sale da 0,79062 a 0,79379, ma resta inferiore al 0,80662 del precedente run
texture crop 720/sigma 4. Il best passa dall'epoca 79 alla 175 e lo stop dalla
99 alla 215: il LR ridotto ha prolungato la fase utile del training.

Il successivo oversampling LAS `70/25/5` migliora rispetto a `50/40/10`
sul test I-AUROC di 0,00512, I-AP di 0,00494, P-AUROC di 0,00290 e
W-P-AUROC di 0,00344. Non migliora invece la precisione di localizzazione:
P-AP varia di -0,00045, W-P-AP di +0,00003 e AUPRO di -0,00069. La P-AP
small scende da 0,07208 a 0,06843. Il limite residuo non sembra quindi la
frequenza degli small sintetici, ma la risoluzione e la ricostruzione spaziale
della score map.

Riportare la griglia da layer2 32x32 a layer3 16x16 migliora sulla validation
I-AUROC di 0,02022 e I-AP di 0,02606, ma sul test I-AUROC resta invariata
(-0,00070). Sul test P-AP cresce di 0,01108 e W-P-AP di 0,00696, mentre
P-AUROC scende di 0,00532, W-P-AUROC di 0,00792 e AUPRO da 0,79310 a
0,77593. La griglia grossolana concentra meglio i punteggi sui pixel anomali
piu evidenti, ma perde copertura e precisione geometrica delle regioni,
soprattutto a basso FPR. Per la localizzazione complessiva layer2 resta quindi
la base preferibile; layer3 e utile come segnale semantico complementare, non
come sostituzione diretta.

L'ablazione post-hoc conferma che sigma 4 sovra-smussa entrambe le griglie.
Su layer2, passando da sigma 4 a sigma 0, P-AP sale di 0,01414, W-P-AP di
0,01480 e AUPRO di 0,00426. Sigma 1 ottiene l'AUPRO migliore (0,82131), ma
rispetto a sigma 0 guadagna solo 0,00045 e perde 0,00065 P-AP: i due valori
sono praticamente equivalenti, con sigma 1 leggermente preferibile se la
priorita e la copertura regionale. Su layer3 sigma 0, 1 e 2 sono quasi
indistinguibili; sigma 4 perde 0,00589 AUPRO rispetto a sigma 0.

Confrontando le migliori varianti di validation, layer3 sigma 0 supera layer2
sigma 0 di 0,00225 P-AUROC, ma perde 0,01533 P-AP, 0,01680 W-P-AP e 0,01511
AUPRO. Per localizzazione, la configurazione candidata resta quindi layer2 con
sigma 1; sigma 0 rimane l'alternativa se si vuole massimizzare strettamente la
Pixel AP.

La valutazione finale di sigma 1 sul test dello stesso checkpoint layer2
conferma la selezione effettuata sulla validation. Rispetto a sigma 4, lascia
invariate I-AUROC/I-AP e migliora P-AUROC di 0,00095, P-AP di 0,01054,
W-P-AUROC di 0,00107 e W-P-AP di 0,01125. Il vantaggio di precisione pixel
generalizza quindi al test e non e un artefatto del solo split di selezione.
Manca soltanto l'AUPRO test a sigma 1 per quantificare anche la copertura
regionale finale.

La prova isolata su `D_upper_detail` produce metriche sensibilmente superiori,
ma non e confrontabile causalmente con posa A: validation e test contengono
solo 61 e 88 immagini, la ruota occupa una porzione diversa dell'inquadratura
e la prevalenza dei pixel anomali test e oltre tre volte quella di posa A
(0,00353 contro 0,00112). Il risultato mostra che TinyGLASS funziona bene sulla
vista di dettaglio, non che la configurazione sia globalmente migliore.

L'aumento dell'input da 256x256 a 384x384, mantenendo posa A, crop 720,
layer2 e sigma 1, e il miglioramento TinyGLASS piu netto finora. Sul test
I-AUROC cresce di 0,04972, I-AP di 0,04066, P-AUROC di 0,02027, P-AP di
0,09756, W-P-AUROC di 0,02250 e W-P-AP di 0,11034. L'AUPRO raggiunge
0,90240; il confronto esatto col precedente sigma 1 non e disponibile, ma
supera di 0,10930 il run 256 sigma 4 dello stesso training.

Il beneficio e presente in tutte le severity sul test. Rispetto al diagnostico
disponibile del precedente layer2 256 `70/25/5` a sigma 4, la P-AP large
passa da 0,24783 a 0,34006, medium da 0,16021 a 0,30467 e small da 0,06843 a
0,15996. Questo confronto per severity include anche il passaggio sigma 4 a 1,
ma l'ablazione globale mostra che lo smoothing spiega solo una parte minore del
salto. La small P-AP cresce quindi di oltre due volte. Il gap I-AUROC
validation-test resta simile al baseline
(0,10832), percio il salto non sembra dovuto soltanto a overfitting sulla
validation. La griglia 48x48 risolve concretamente il precedente limite di
localizzazione della griglia 32x32.

Sul run 384, passare da layer2 48x48 a layer3 24x24 riduce le patch da 2304 a
576. La parte patch-wise del discriminatore, GAS e la validation feature cache
si riduce quindi del 75%. Con 128 canali float32 e 392 immagini di validation,
la sola cache teorica passa da circa 441 MiB a 110 MiB. Il backbone elabora
comunque l'intero input 384x384 e produce sia layer2 sia layer3, quindi memoria
e tempo end-to-end non si riducono di quattro volte; lo speedup reale va
misurato dai tempi per epoca.

Il risparmio ha un costo moderato ma sistematico sul test: rispetto a layer2
sigma 1, layer3 sigma 1 perde 0,00613 I-AUROC, 0,00077 P-AUROC, 0,01300 P-AP,
0,00301 W-P-AUROC, 0,02402 W-P-AP e 0,00572 AUPRO; I-AP cresce invece di
0,00760. A FPR 0,05 l'AUPRO scende da 0,73927 a 0,72560. Per severity, layer3
migliora la P-AP large ma peggiora medium e small. L'ablazione seleziona sigma
4 per layer3, ma sulla validation resta sotto layer2 sigma 1 di 0,02068
W-P-AP e 0,00269 AUPRO. Layer2 resta pertanto il preset di qualita; layer3 e
un valido preset efficiente quando cache, memoria GAS o tempo di training sono
il vincolo principale.

L'ulteriore aumento da 384 a 512 con layer2 porta la griglia da 48x48 a 64x64
ma regredisce su tutte le metriche globali principali. Sul test perde 0,03717
I-AUROC, 0,02358 I-AP, 0,00524 P-AUROC, 0,01928 P-AP, 0,00844 W-P-AUROC,
0,03070 W-P-AP e 0,00974 AUPRO rispetto al 384 layer2 sigma 1. A FPR 0,05
l'AUPRO scende di 0,02822.

La regressione deriva soprattutto da falsi positivi normali: la mediana dello
score clean test sale da 0,24155 a 0,40800, mentre quella hole cresce solo da
0,55127 a 0,58376. La P-AP small migliora leggermente da 0,15996 a 0,16651,
ma medium scende da 0,30467 a 0,28117 e large da 0,34006 a 0,30911. Il calo e
particolarmente evidente su illuminazione dusty e wear light/evident, segnale
che la risoluzione aggiuntiva rende il discriminatore piu sensibile a texture
e variazioni normali. Sigma 4 e il migliore per il checkpoint 512 e recupera
parte della localizzazione, ma sulla validation resta sotto il 384 sigma 1 di
0,01483 P-AP, 0,03780 W-P-AP e 0,01611 AUPRO. La curva di scala ha quindi un
massimo osservato a 384x384; 512 non sostituisce il baseline.

Il run 512 layer3 corregge gran parte della regressione di 512 layer2. A parita
di input, portare la griglia da 64x64 a 32x32 migliora sul test I-AUROC di
0,03524, I-AP di 0,04202, P-AUROC di 0,01104, P-AP di 0,03457,
W-P-AUROC di 0,01161, W-P-AP di 0,03178 e AUPRO di 0,02498. La mediana clean
scende da 0,40800 a 0,31644: la griglia meno densa elimina parte della
sensibilita a texture normali senza perdere il dettaglio acquisito dal
backbone a 512.

Il comportamento per scala dipende quindi dal layer. Con layer2, le metriche
test crescono da 256 a 384 e poi calano a 512; con layer3 crescono invece in
modo monotono da 256 a 384 e 512. Il 512 layer3 sigma 1 supera sul test il 384
layer2 sigma 1 di 0,00580 P-AUROC, 0,01529 P-AP, 0,00318 W-P-AUROC, 0,00109
W-P-AP e 0,01524 AUPRO, con I-AUROC quasi invariata (-0,00193). Sulla
validation il quadro e misto: 384 layer2 conserva AUPRO piu alta, mentre 512
layer3 ha P-AP piu alta. L'ablazione sceglie sigma 4 per 512 layer3; rispetto a
384 layer2 sigma 1 pareggia sostanzialmente W-P-AP (+0,00026), aumenta P-AP
di 0,02795 ma perde 0,00619 AUPRO.

In termini di costo, 512 layer3 usa 1024 patch, contro 2304 del 384 layer2 e
4096 del 512 layer2: il carico patch-wise e rispettivamente inferiore del
55,6% e del 75%. La cache teorica su 392 immagini e circa 196 MiB, contro 441
MiB e 784 MiB. Il backbone 512 resta tuttavia il 77,8% piu costoso in pixel
rispetto al backbone 384, quindi il vantaggio riguarda soprattutto
discriminatore, GAS e cache, non necessariamente la latenza totale.

La valutazione test del sigma 4 selezionato sulla validation conferma il
vantaggio rispetto a sigma 1 sullo stesso checkpoint 512 layer3: P-AUROC
cresce di 0,00131, P-AP di 0,00586, W-P-AUROC di 0,00143, W-P-AP di 0,00543
e AUPRO di 0,00248. Rispetto al 384 layer2 sigma 1, il 512 layer3 sigma 4
mantiene I-AUROC quasi invariata (-0,00193) e migliora P-AUROC di 0,00711,
P-AP di 0,02116, W-P-AUROC di 0,00460, W-P-AP di 0,00652 e AUPRO di
0,01772. Questi confronti test sono descrittivi: dopo molte configurazioni
osservate, il test corrente non rappresenta piu un holdout incontaminato.

## SuperSimpleNet

| Esperimento | Risultati validation | Risultati test |
|---|---|---|
| Baseline, input 256x256, 100 epoche, massimo 1.024 immagini/epoca, batch 16, sola posa `A_overhead`, pose crop attivo | I-AUROC 0,84072; I-AP 0,66665; P-AUROC 0,98585; P-AP 0,18444; W-P-AUROC 0,98441; W-P-AP 0,18641; AUPRO 0,91505 | I-AUROC 0,69737; I-AP 0,72447; P-AUROC 0,97511; P-AP 0,19493; W-P-AUROC 0,97246; W-P-AP 0,19729; AUPRO 0,88977 |
| Estensione della baseline a 200 epoche, input 256x256, massimo 1.024 immagini/epoca, batch 16, sola posa `A_overhead`, pose crop attivo | I-AUROC 0,74546; I-AP 0,54837; P-AUROC 0,98237; P-AP 0,20811; W-P-AUROC 0,98102; W-P-AP 0,21073; AUPRO 0,90902 | I-AUROC 0,64643; I-AP 0,67704; P-AUROC 0,97039; P-AP 0,20621; W-P-AUROC 0,96783; W-P-AP 0,21007; AUPRO 0,87194 |
| Posa `A_overhead`, pose crop attivo, input 384x384, massimo 200 epoche; epoca selezionata non inclusa nel report | I-AUROC 0,70710; I-AP 0,49844; P-AUROC 0,98019; P-AP 0,17165; W-P-AUROC 0,97883; W-P-AP 0,17646; AUPRO 0,91986 | I-AUROC 0,64771; I-AP 0,65914; P-AUROC 0,97695; P-AP 0,16881; W-P-AUROC 0,97523; W-P-AP 0,17630; AUPRO 0,91593 |
| Stessa configurazione della baseline, sola posa `A_overhead`, pose crop disattivato | I-AUROC 0,77643; I-AP 0,58534; P-AUROC 0,97840; P-AP 0,09183; W-P-AUROC 0,96204; W-P-AP 0,09642; AUPRO 0,88463 | I-AUROC 0,65263; I-AP 0,66451; P-AUROC 0,95874; P-AP 0,07268; W-P-AUROC 0,92714; W-P-AP 0,07491; AUPRO 0,83399 |
| Stessa configurazione della baseline, tutte le pose, pose crop disattivato | I-AUROC 0,56630; I-AP 0,33248; P-AUROC 0,94970; P-AP 0,08772; W-P-AUROC 0,92576; W-P-AP 0,09368; AUPRO 0,75477 | I-AUROC 0,54051; I-AP 0,54791; P-AUROC 0,93718; P-AP 0,08331; W-P-AUROC 0,90641; W-P-AP 0,08910; AUPRO 0,74092 |
| Tutte le pose, pose crop attivo, input 256x256; epoca selezionata non inclusa nel report | I-AUROC 0,59152; I-AP 0,34197; P-AUROC 0,93082; P-AP 0,11619; W-P-AUROC 0,92324; W-P-AP 0,12047; AUPRO 0,71472 | I-AUROC 0,54950; I-AP 0,55512; P-AUROC 0,91680; P-AP 0,11397; W-P-AUROC 0,90656; W-P-AP 0,11909; AUPRO 0,69995 |
| Rivalutazione del checkpoint baseline 256x256 con score immagine fissato alla media del top 1% della mappa; nessun retraining | I-AUROC 0,92071; I-AP 0,84453; P-AUROC 0,98585; P-AP 0,18444; W-P-AUROC 0,98441; W-P-AP 0,18641; AUPRO 0,91505 | I-AUROC 0,81105; I-AP 0,84723; P-AUROC 0,97511; P-AP 0,19493; W-P-AUROC 0,97246; W-P-AP 0,19729; AUPRO 0,88977 |
| Nuovo training 256x256, posa A, crop, anomalie Perlin limitate alla target wheel, soglia 0,6 e score top 1% pre-registrato; training collassato | I-AUROC 0,54691; I-AP 0,27143; P-AUROC 0,73058; P-AP 0,00195; W-P-AUROC 0,65978; W-P-AP 0,00196; AUPRO 0,39845 | I-AUROC 0,55728; I-AP 0,53760; P-AUROC 0,68904; P-AP 0,00285; W-P-AUROC 0,60744; W-P-AP 0,00286; AUPRO 0,39243 |
| Nuovo training 256x256, posa A, crop, anomalie Perlin limitate alla target wheel, soglia 0,2 e score top 1% pre-registrato | I-AUROC 0,94074; I-AP 0,86440; P-AUROC 0,98376; P-AP 0,15428; W-P-AUROC 0,97976; W-P-AP 0,15453; AUPRO 0,92189 | I-AUROC 0,83133; I-AP 0,86214; P-AUROC 0,97317; P-AP 0,16779; W-P-AUROC 0,96659; W-P-AP 0,16834; AUPRO 0,88153 |
| Ablazione post-hoc sigma `0/1/2/4` sul checkpoint Perlin 0,2; selezione con W-P-AP validation | Sigma 4 selezionato: P-AUROC 0,98376; P-AP 0,15428; W-P-AUROC 0,97976; W-P-AP 0,15453; AUPRO 0,92189 | Solo sigma 4: P-AUROC 0,97317; P-AP 0,16779; W-P-AUROC 0,96659; W-P-AP 0,16834; AUPRO 0,88153 |
| Training 256x256, posa A, crop, Perlin 0,2 senza restrizione alla target wheel, score top 1% e sigma 4 | I-AUROC 0,93349; I-AP 0,86656; P-AUROC 0,98492; P-AP 0,18342; W-P-AUROC 0,98342; W-P-AP 0,18490; AUPRO 0,90371 | I-AUROC 0,82722; I-AP 0,86083; P-AUROC 0,96967; P-AP 0,19104; W-P-AUROC 0,96657; W-P-AP 0,19298; AUPRO 0,87530 |
| Ablazione post-hoc sigma `0/1/2/4` sul checkpoint Perlin 0,2 senza restrizione; selezione con W-P-AP validation | Sigma 4 selezionato: P-AUROC 0,98492; P-AP 0,18342; W-P-AUROC 0,98342; W-P-AP 0,18490; AUPRO 0,90371 | Solo sigma 4: P-AUROC 0,96967; P-AP 0,19104; W-P-AUROC 0,96657; W-P-AP 0,19298; AUPRO 0,87530 |

Benchmark su Tesla T4, batch 1 e input 256x256: predizione media 24,71 ms,
end-to-end medio 67,67 ms, throughput 40,47 immagini/s e picco incrementale
allocato 81,00 MiB. Questo run costituisce la baseline SuperSimpleNet corrente.
Nella rivalutazione dello stesso checkpoint con top 1%, la predizione media e
24,60 ms, l'end-to-end medio 59,51 ms, il throughput 40,66 immagini/s e il
picco incrementale allocato resta 81,00 MiB.
Sul checkpoint Perlin 0,2, sempre su Tesla T4 e batch 1, la predizione media e
24,40 ms, l'end-to-end medio 65,36 ms e il picco incrementale allocato e
81,00 MiB. Il throughput riportato di 40,99 immagini/s riguarda la sola fase
host-to-device e predict; includendo decode e preprocessing, il throughput
sequenziale medio e circa 15,30 immagini/s.

L'ablazione Gaussian sigma del checkpoint Perlin 0,2 mostra un miglioramento
monotono della localizzazione da sigma 0 a sigma 4. Rispetto a sigma 0,
sigma 4 aumenta sulla validation P-AUROC di 0,00901, P-AP di 0,02207,
W-P-AUROC di 0,01110, W-P-AP di 0,02192 e AUPRO di 0,02606. Le metriche
image-level rimangono esattamente invariate, come richiesto dal top 1%
calcolato sulla griglia nativa. Sigma 4 resta quindi il valore confermato.

Disattivare la restrizione delle anomalie sintetiche alla ruota produce il
miglior compromesso SuperSimpleNet osservato. Rispetto al run ristretto, sul
test perde solo 0,00412 I-AUROC e 0,00132 I-AP, ma guadagna 0,02325 P-AP e
0,02464 W-P-AP; W-P-AUROC resta invariata (-0,00002). Il guadagno P-AP e
presente per tutte le severity: +0,05199 large, +0,01740 medium e +0,01297
small. AUPRO scende di 0,00623, indicando mappe piu precise sui pixel positivi
ma leggermente meno complete come copertura regionale. Anche senza restrizione
l'ablazione sigma seleziona monotonicamente sigma 4.
Il confronto fattoriale tra posa A/tutte le pose e crop attivo/disattivato e
ora completo; per il run tutte-pose con crop manca nel report il riepilogo del
training necessario a verificare il protocollo early stopping.


