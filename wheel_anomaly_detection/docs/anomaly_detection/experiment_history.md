# Experiment history

Metriche: `I-AUROC` = Image AUROC, `I-AP` = Image AP, `P-AUROC` = Pixel AUROC,
`P-AP` = Pixel AP, `W-P-AUROC`/`W-P-AP` = metriche pixel limitate alla ruota.

I nuovi risultati forniti dall'utente vengono sempre registrati qui e nei
rispettivi artefatti locali, senza richiedere una conferma ogni volta.
Conservare i report originali e la provenienza; dati mancanti o riepiloghi
separati non vanno trasformati in misure complete o valori ricostruiti.

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
| Reference, sola posa `A_overhead`, crop 720x720, input 224x224, max train 1.000.000, bank 50.000, sigma 4; run 2026-10-06 | I-AUROC 0,96432; I-AP 0,91836; P-AUROC 0,99674; P-AP 0,40763; W-P-AUROC 0,99654; W-P-AP 0,41202; AUPRO 0,96682 | I-AUROC 0,86428; I-AP 0,88631; P-AUROC 0,99429; P-AP 0,42891; W-P-AUROC 0,99402; W-P-AP 0,43602; AUPRO 0,95442 |
| Light, sola posa `A_overhead`, crop 720x720, input 384x512, max train 1.000.000, bank 50.000, sigma 4; run `20261006_185850` | I-AUROC 0,91609; I-AP 0,78478; P-AUROC 0,99845; P-AP 0,45681; W-P-AUROC 0,99831; W-P-AP 0,48612; AUPRO 0,99206 | I-AUROC 0,79283; I-AP 0,77494; P-AUROC 0,99783; P-AP 0,41054; W-P-AUROC 0,99772; W-P-AP 0,45210; AUPRO 0,99103 |

Run posa A: `patchcore_reference_wide_resnet50_2_224x224_A_crop720_bank50k__seed42__20261006_122506`.
Training su 2.800 immagini; validation 296 clean + 96 hole; test 404 clean +
404 hole. Il fit ha richiesto 646,05 s. Manifest SHA-256:
`afc4d44cf8330b94c52f21a314b797abdc553d517348b11489ac15e32a3c06fb`.
Metriche a precisione completa, diagnostiche per severity/lighting/wear e
benchmark sono conservati nel
[record locale PatchCore](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/metrics.json).
Questo run va distinto dai Reference storici: posa, crop e contratto pixel
non sono un'ablazione controllata della sola memory bank.

Nuovo Light posa A:
`patchcore_light_resnet18_384x512_A_crop720_bank50k__seed42__20261006_185850`.
ResNet-18, layer2/layer3, embedding 384, griglia 48x64,
`num_neighbors=9`, reservoir 1M, bank effettiva 50k (73,2422 MiB).
Fit: 398,6082 s su 2.800 immagini; validation 296 clean + 96 hole,
test 404 clean + 404 hole. Manifest identico al Reference posa A.
[Metriche complete Light](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/metrics.json),
[diagnostiche estese](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/extended_diagnostics.json).
Le due metriche incollate sono associate rispettivamente a validation/test:
I-AUROC, I-AP e P-AP coincidono esattamente con i gruppi posa A delle
diagnostiche. La saturazione degli score normalizzati è nulla su entrambi
gli split. AUPRO test ai cutoff 0,05/0,10/0,30:
0,94927 / 0,97353 / 0,99103, con 406 regioni.

Questo Light non sostituisce i run storici multiposa. Anche il confronto
pixel con Reference posa A ha contratti diversi: Light ridimensiona il
crop 720 a 384x512, Reference usa resize lato corto 256 e center crop 224.
Sul test la frazione di pixel anomali Light è 0,0011199095, contro
0,00141369 del Reference, e le regioni sono 406 contro 411.
Quindi i cambiamenti di AP/AUPRO non sono un'ablazione della sola backbone.

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

## Benchmark di efficienza importati — 2026-10-06 e 2026-10-07

Fonte: report forniti dall'utente in chat; registrazione locale, senza nuova
esecuzione o ricalcolo delle metriche. I JSON completi mantengono i valori
originali, ambiente software, configurazioni e provenienza. Gli output sono
locali sotto `outputs/anomaly_detection/` e ignorati da Git.

Protocollo comune: Tesla T4, FP32, batch 1, split validation della sola posa
`A_overhead`, crop `(top=90, left=240, height=720, width=720)`, 10 warm-up e
50 misure per scope. Il manifest coincide:
`afc4d44cf8330b94c52f21a314b797abdc553d517348b11489ac15e32a3c06fb`.
Loader sequenziale, zero worker, senza prefetch.

`predict_only` misura `detector.predict` su una singola immagine gia
preprocessata e residente in GPU; `host_to_device_and_predict` include anche
il trasferimento. `end_to_end` include il loader, che legge e trasforma anche
maschere/annotazioni: non rappresenta una pipeline di deployment senza
maschere. Le finestre temporali sono separate; per PatchCore la media
host-to-device + predict (58,3842 ms) e inferiore alla media predict-only
(62,8859 ms) nel nuovo run con energia. Si conservano entrambe senza sommarle o ricostruire
un tempo di trasferimento per differenza.

| Modello / input | Predict-only medio (ms) | Predict-only p95 (ms) | H2D + predict medio (ms) | End-to-end medio (ms) | FPS predict-only | FPS H2D + predict | FPS end-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| PatchCore Reference, 224x224; nuovo fit con energia | 62,8859 | 66,3945 | 58,3842 | 120,1384 | 15,9018 | 17,1279 | 8,3237 |
| PatchCore Light, 384x512, posa A, bank 50k | 101,2088 | 103,6088 | 100,2939 | 140,7615 | 9,8806 | 9,9707 | 7,1042 |
| TinyGLASS TG-384, 384x384, layer2, sigma 1; misura 2026-10-07 | 5,1284 | 5,9211 | 5,6582 | 62,4856 | 194,9935 | 176,7337 | 16,0037 |
| TinyGLASS, 512x512, layer3, sigma 4; nuova misura con energia | 6,2933 | 8,9416 | 8,3195 | 62,9831 | 158,9000 | 120,1996 | 15,8773 |
| EfficientAD-S, 256x256, global_max | 57,7160 | 58,5829 | 55,3607 | 99,7747 | 17,3262 | 18,0633 | 10,0226 |
| SuperSimpleNet, 256x256, Perlin 0,2 unrestricted, top-1%, sigma 4 | 27,5438 | 31,8189 | 26,3263 | 78,9500 | 36,3058 | 37,9848 | 12,6662 |

| Risorsa / scope | PatchCore Reference | PatchCore Light | TinyGLASS TG-384 | TinyGLASS TG-512 | EfficientAD-S | SuperSimpleNet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parametri totali | 24.862.528 | 2.782.784 | 2.850.368 | 2.850.368 | 8.057.856 | 33.719.363 |
| Parametri con requires_grad | 0 | 0 | 67.584 | 67.584 | 5.363.712 | 8.856.835 |
| Parametri (MiB) | 94,8430 | 10,6155 | 10,8733 | 10,8733 | 30,7383 | 128,6292 |
| Buffer (MiB; includono la bank) | 195,4637 | 73,2594 | 0,0216 | 0,0216 | 0,0694 | 0,1600 |
| Memory bank (MiB; gia inclusa nei buffer) | 195,3125 | 73,2422 | Assente | Assente | Assente | Assente |
| Checkpoint (MiB) | 290,3828 | 83,9034 | 11,0202 | 10,9876 | 30,8275 | 128,8735 |
| CUDA baseline allocata, predict-only (MiB) | 301,0288 | 191,2793 | 20,7207 | 23,0332 | 31,5645 | 139,6885 |
| CUDA picco allocato, predict-only (MiB) | 385,2476 | 277,5293 | 38,7207 | 55,0332 | 878,4482 | 219,6885 |
| CUDA incremento picco allocato, predict-only (MiB) | 84,2188 | 86,2500 | 18,0000 | 32,0000 | 846,8838 | 80,0000 |
| CUDA incremento picco allocato, pipeline (MiB) | 84,7930 | 88,5000 | 19,6875 | 35,0000 | 847,6338 | 80,7500 |
| CUDA picco riservato, predict-only/pipeline (MiB) | 602,0000 | 712,0000 | 46,0000 | 78,0000 | 970,0000 | 254,0000 |
| Energia GPU predict-only (J/image) | 4,1157 (3 finestre complete) | 6,8485 (3 finestre complete) | 0,2394 (3 finestre complete) | 0,3686 (3 finestre complete) | 4,0317 (log) | 1,9389 (3 finestre complete) |

La memoria CUDA e quella dell'allocator e comprende modello, memory bank e
altri tensori vivi sul device; non include le allocazioni del driver. La bank
Reference è `50000x1024` FP32 (195,31 MiB); quella Light è
`50000x384` FP32 (73,24 MiB). Entrambe sono già incluse nei buffer
e nel picco CUDA, non vanno sommate una seconda volta.
Il picco allocato comprende tutti i tensori vivi nella sessione, non solo
il modello; quello riservato è riportato separatamente e non viene
reinterpretato come memoria allocata.
Nei primi report dei tre modelli
l'energia era `null`; questi originali restano conservati. Per PatchCore e
TinyGLASS la tabella usa ora i nuovi benchmark con tre finestre energetiche
complete. PatchCore è stato rifittato: l'efficienza della tabella appartiene
al nuovo run `20261006_184126`, non al run di accuratezza `20261006_122506`.
Per EfficientAD-S è disponibile un riepilogo energetico separato fornito in
chat, documentato qui sotto.
La latenza non viene convertita in una stima energetica.

### Provenienza PatchCore

Run `patchcore_reference_wide_resnet50_2_224x224_A_crop720_bank50k__seed42__20261006_122506`,
Wide-ResNet50-2, layer2/layer3, embedding 1.024, sigma 4. Il reservoir contiene
1.000.000 candidati e la bank effettiva 50.000 embedding.
[Benchmark completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/inference_benchmark_batch1.json),
[metriche validation/test](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/metrics.json),
[diagnostiche estese](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/extended_diagnostics.json).
Le metriche di accuratezza sono quelle del nuovo run posa A riportato nella
sezione PatchCore, non quelle dei vecchi run multiposa.

#### Nuovo fit PatchCore Reference con energia — 2026-10-06

Run `patchcore_reference_wide_resnet50_2_224x224_A_crop720_bank50k__seed42__20261006_184126`.
Il nuovo fit ha richiesto **627,3508 s** su 2.800 immagini.
Configurazione, manifest, posa, crop, input, forma della bank e riepilogo
del fit coincidono con il precedente run `20261006_122506`: Wide-ResNet50-2,
layer2/layer3, input 224x224, crop 720, embedding 1.024, reservoir 1M,
bank 50k e sigma 4. Questa corrispondenza non verifica l'identità dei
tensori della bank: non è stato fornito un checksum del checkpoint.
L'allegato contiene solo benchmark e misura energetica, non nuove metriche
di accuratezza; quelle del fit precedente restano associate al proprio run.

[Benchmark completo del nuovo run](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/run_20261006_184126/inference_benchmark_batch1.json),
[energia con finestre e contatori NVML](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/run_20261006_184126/inference_energy_batch1.json),
[console originale](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/run_20261006_184126/user_console_output.txt),
[provenienza dell'import](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_reference_pose_a/run_20261006_184126/import_provenance.json).

Tesla T4, FP32, batch 1, 10 warm-up e 50 misure per scope:
predict-only **62,8859 ms**, p95 **66,3945 ms**, **15,9018 immagini/s**;
H2D + predict **58,3842 ms**; end-to-end **120,1384 ms**.
Picco CUDA allocato **385,2476 MiB**, incremento predict-only
**84,2188 MiB**, bank **195,3125 MiB**, checkpoint **290,3828 MiB**.
La fase di decode/preprocessing misura 61,7542 ms nel nuovo run,
contro 40,8436 ms nel precedente: l'end-to-end include anche questo lavoro,
senza rappresentare un cambiamento della sola architettura.

Energia misurata tramite `nvml_total_energy_counter`, warm-up effettivo
5,0614 s, tre finestre complete da circa 30 s e nessun warning:

| Ripetizione | Inferenze | Durata (s) | Energia GPU (J) | J/image | Potenza media (W) |
|---|---:|---:|---:|---:|---:|
| 1 | 481 | 30,0533 | 1.983,563 | 4,123832 | 66,001501 |
| 2 | 483 | 30,0253 | 1.989,161 | 4,118346 | 66,249422 |
| 3 | 483 | 30,0206 | 1.982,762 | 4,105097 | 66,046715 |

Aggregato originale: **4,1157470629 J/image**, potenza media
**66,0991852137 W**, deviazione standard fra le medie delle finestre
**0,0096314551 J/image**. Totale: 1.447 inferenze, 90,099235 s,
5.955,486 J. Contatori in mJ, somme, aggregati ponderati e dispersione
sono coerenti con il report. La deviazione standard non descrive
l'energia delle singole inferenze, che non viene misurata individualmente.

Lo scope è predict-only, input residente in GPU; CPU, loader e
trasferimento dell'input sono esclusi. Il valore riguarda l'intera GPU,
senza attribuzione al processo e senza sottrazione dell'idle.
CUDA/NVML sono associati per UUID; un solo processo compute rilevato,
power limit 70 W. La fase energetica non esegue fit o calibrazione,
ma il run complessivo ha effettuato il nuovo fit riportato sopra.

### Provenienza PatchCore Light posa A

Run `patchcore_light_resnet18_384x512_A_crop720_bank50k__seed42__20261006_185850`;
fase energetica completata il `2026-10-06T19:09:18.567611+00:00`.
Tesla T4, FP32, batch 1, sigma 4, input 384x512, posa A,
crop 720, ResNet-18, embedding 384 e bank 50k.
La griglia di query è 48x64 (3.072 patch), contro 28x28 (784)
del Reference; cambia anche `num_neighbors`, da 1 a 9.

[Benchmark completo Light](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/inference_benchmark_batch1.json),
[energia con contatori NVML](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/inference_energy_batch1.json),
[metriche validation/test](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/metrics.json),
[diagnostiche complete](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/extended_diagnostics.json),
[console](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/user_console_output.txt),
[provenienza](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/patchcore_light_pose_a/run_20261006_185850/import_provenance.json).
Le metriche sono trascritte dai due dizionari forniti in chat; le diagnostiche
provengono dal secondo allegato, senza ricalcolo.

Energia: **6,8485224215 J/image**, potenza media **67,8032546417 W**,
deviazione standard tra le medie delle finestre **0,0041311614 J/image**.
Tre finestre complete da circa 30 s via `nvml_total_energy_counter`,
warm-up effettivo 5,0700 s, nessun warning:

| Ripetizione | Inferenze | Durata (s) | Energia GPU (J) | J/image | Potenza media (W) |
|---|---:|---:|---:|---:|---:|
| 1 | 297 | 30,0065 | 2.032,932 | 6,844889 | 67,749817 |
| 2 | 297 | 30,0329 | 2.033,752 | 6,847650 | 67,717447 |
| 3 | 298 | 30,0578 | 2.042,198 | 6,853013 | 67,942338 |

Totale: 892 inferenze, 90,097179 s e 6.108,882 J.
Delta dei contatori, aggregati ponderati e dispersione verificati contro il
report. Stesso scope energetico degli altri modelli: intera GPU,
input residente, esclusi CPU/loader/transfer, nessuna sottrazione dell'idle.
Un solo processo compute rilevato; UUID CUDA/NVML coincidente con la
sessione del nuovo Reference e power limit 70 W.

Light riduce il picco allocato del 28,0% e la bank del 62,5% rispetto
al Reference, ma in queste configurazioni aumenta la latenza predict-only
del 60,9% e l'energia per immagine del 66,4%.
La backbone più piccola non garantisce un'inferenza complessiva più economica:
risoluzione, numero di query, dimensione degli embedding e vicini cambiano
insieme. Il maggior numero di query è una possibile spiegazione del costo
osservato, non una causa isolata verificata tramite profiling o ablation.

Nota da riprendere nella discussione dei risultati: compromesso tra
risoluzione dell'input, dettaglio dei difetti e costo energetico.
Per localizzare piccoli fori occorre conservarne una rappresentazione
spaziale sufficiente nell'input; un ridimensionamento eccessivo può perdere
dettagli utili. Aumentare la risoluzione è una possibile scelta per ottenere
una localizzazione migliore, ma nei modelli patch-based aumenta anche il
numero di query alla memory bank e può richiedere più tempo ed energia
per immagine. Una backbone Light non elimina questo costo: il confronto
misurato mostra meno memoria, ma maggiore latenza ed energia.
Nei risultati va quindi valutato il compromesso qualità-efficienza
dell'intera configurazione, non soltanto la dimensione della backbone.
Non presentare questi run come prova che immagini più grandi siano sempre
necessarie o sufficienti per buoni risultati: cambiano anche backbone,
embedding, scoring e geometria delle maschere; per isolare il beneficio
della risoluzione serve un'ablazione a parità degli altri fattori.

### Provenienza TinyGLASS

Benchmark creato il `2026-10-06T14:16:21.750245+00:00` sul checkpoint
`tinyglass_A_overhead_crop720_input384_layer2_sigma1_hole_small70_lowlr_pat40_frozen__seed42__20260901_102247`,
Drive file ID `1gq7zyfAJZ5Vza2GjLBsIA91lVKU-b3LW`.
Il nome del run contiene `input384_layer2_sigma1`, ma il checkpoint e il
benchmark verificano input **512x512** e griglia **layer3** (32x32).
Lo smoothing salvato era sigma 1; sigma **4** e stato applicato dopo restore
stretto, come selezionato nell'ablazione locale preesistente.

[Benchmark storico TinyGLASS, senza energia](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/tinyglass_512_layer3_sigma4/inference_benchmark_batch1.json).
Questo notebook non ha eseguito training, selezione sigma o una nuova
valutazione di accuratezza. Per l'accuratezza resta valido il risultato gia
registrato per 512 layer3 sigma 4: sul test I-AUROC 0,77330, I-AP 0,81126,
P-AUROC 0,97924, P-AP 0,27592, W-P-AUROC 0,97646, W-P-AP 0,28299 e
AUPRO@0,30 0,92012.

#### Aggiornamento TinyGLASS: benchmark con energia — 2026-10-06

Il nuovo benchmark è stato creato il `2026-10-06T16:56:58.864672+00:00`;
la fase energetica è stata completata il
`2026-10-06T16:58:33.956302+00:00`. Stesso checkpoint finale e medesima
configurazione effettiva: input 512x512, layer3, posa A, crop 720,
sigma 4 applicato dopo restore stretto. Manifest coincidente; nessun nuovo
training, selezione sigma o test di accuratezza.

[Nuovo benchmark completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/tinyglass_512_layer3_sigma4/efficiency_20261006_165658/inference_benchmark_batch1.json),
[report energetico completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/tinyglass_512_layer3_sigma4/efficiency_20261006_165658/inference_energy_batch1.json),
[output console fornito dall'utente](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/tinyglass_512_layer3_sigma4/efficiency_20261006_165658/user_console_output.txt),
[provenienza dell'import](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/tinyglass_512_layer3_sigma4/efficiency_20261006_165658/import_provenance.json).
Il report storico non è stato sovrascritto: la sua precedente latenza
predict-only era 6,8684 ms, contro 6,2933 ms nella nuova misura. Sono due
misurazioni, non un cambiamento del modello.

Energia misurata con `nvml_total_energy_counter`, Tesla T4, FP32,
batch 1, input già residente in GPU: warm-up effettivo 5,0033 s e
3 finestre complete da circa 30 s, senza warning.

| Ripetizione | Inferenze | Durata (s) | Energia GPU (J) | J/image | Potenza media (W) |
|---|---:|---:|---:|---:|---:|
| 1 | 5.634 | 30,0029 | 2.034,665 | 0,361140 | 67,815694 |
| 2 | 5.467 | 30,0046 | 2.014,290 | 0,368445 | 67,132629 |
| 3 | 5.261 | 30,0051 | 1.982,132 | 0,376760 | 66,059896 |

Aggregato originale: **0,3686032881 J/image**, potenza media
**67,0027189497 W**, deviazione standard fra le medie delle finestre
**0,0078150124 J/image**. Totale: 16.362 inferenze, 90,012571 s,
6.031,087 J. I delta dei contatori in mJ e gli aggregati ponderati sono stati
controllati contro il JSON, senza sostituirne i valori con stime.
La dispersione riportata non è la deviazione standard dell'energia delle
singole immagini, che non viene misurata individualmente.

Lo scope è l'intera GPU, non il singolo processo o il sistema completo;
CPU, loader e trasferimento dell'input sono esclusi, il consumo a riposo
non viene sottratto. CUDA e NVML sono associati per UUID; il report rileva
un solo processo compute e un power limit di 70 W. Le finestre energetiche
sono separate da quelle della latenza: non si ricava energia moltiplicando
la latenza breve per una potenza nominale.
Il JSON console precede il salvataggio finale e non contiene
`benchmark_run_id`: nessun identificativo mancante è stato inventato.

#### TG-512: sottogruppi del test a sigma 4 — registrati il 2026-10-07

Il riepilogo fornito dall'utente riguarda il test TG-512 del notebook:
512x512, layer3, sigma 4, 404 coppie clean/hole posa A. Per ciascuno dei
quattro campi i conteggi ricostituiscono 404 clean e 404 hole. Le percentuali
sono quelle stampate a due decimali; il riepilogo non sostituisce i JSON
diagnostici completi e non contiene prevalenza pixel o hash del nuovo test.

| Campo | Gruppo | Clean | Hole | I-AUROC (%) | I-AP (%) | P-AP (%) |
|---|---|---:|---:|---:|---:|---:|
| severity | large | 39 | 39 | 87,18 | 88,99 | 35,13 |
| severity | medium | 157 | 157 | 79,11 | 82,84 | 32,98 |
| severity | small | 208 | 208 | 74,09 | 78,18 | 21,13 |
| camera_pose | A_overhead | 404 | 404 | 77,33 | 81,12 | 27,59 |
| lighting | mars_clear_refined | 97 | 97 | 81,29 | 85,43 | 33,19 |
| lighting | mars_dusty_refined | 307 | 307 | 77,24 | 80,46 | 26,66 |
| wear | surface_current | 85 | 85 | 79,33 | 82,12 | 28,78 |
| wear | wear_evident | 140 | 140 | 76,69 | 81,28 | 29,24 |
| wear | wear_light | 179 | 179 | 77,42 | 81,01 | 26,46 |

AUPRO originale, scala frazionaria: @0,05 **0,7654463871116568**;
@0,10 **0,8356311439858932**; @0,30 **0,9201249772458705**.

[Riepilogo JSON](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_512_layer3_sigma4/test_sigma4_user_summary/test_user_summary.json),
[sottogruppi CSV in percentuale](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_512_layer3_sigma4/test_sigma4_user_summary/test_subgroup_metrics_percent.csv),
[testo della chat](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_512_layer3_sigma4/test_sigma4_user_summary/user_subgroup_output.txt),
[provenienza](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_512_layer3_sigma4/test_sigma4_user_summary/import_provenance.json).

#### TG-384: benchmark con energia — 2026-10-07

Report fornito dall'utente: benchmark creato il `2026-10-07T09:37:11.089105+00:00`,
fase energetica completata il `2026-10-07T09:38:46.175796+00:00`.
Checkpoint finale `20260831_112751`, Drive ID
`1L45AfJ4ZeFceXfy2houemOVc_3Kv4Qpr`: input **384x384**, **layer2** (48x48),
**sigma 1** conservato dopo restore, ResNet-18 congelata e LAS hole 70/25/5.
SHA-256 checkpoint `df576d6773d7f2ecb05372facf9cd916788ac31a66e39e8778b849653d9fc270`,
coincidente con il run manifest verificato durante la preparazione del
notebook. Manifest canonico coincidente, validation posa A con 296 clean
e 96 hole. Nessun training, selezione sigma o test di accuratezza.

Protocollo e ambiente coincidono con il precedente TG-512: Tesla T4, FP32,
batch 1, 10 warm-up e 50 misure per scope, stesso crop e loader, Python
3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8 e cuDNN 91900.

| Risorsa / scope | TG-384: 384/layer2/sigma1 | TG-512: 512/layer3/sigma4 |
|---|---:|---:|
| Predict-only medio (ms) | 5,128375 | 6,293266 |
| Predict-only p95 (ms) | 5,921135 | 8,941595 |
| H2D + predict medio (ms) | 5,658231 | 8,319499 |
| End-to-end medio (ms) | 62,485621 | 62,983103 |
| CUDA picco allocato, predict-only (MiB) | 38,720703 | 55,033203 |
| CUDA incremento picco allocato, predict-only (MiB) | 18,000000 | 32,000000 |
| Energia GPU predict-only (J/image) | 0,239375737 | 0,368603288 |

Energia misurata tramite `nvml_total_energy_counter`, warm-up effettivo
5,002830 s, tre finestre complete da circa 30 s, nessun warning, un solo
processo compute e associazione CUDA/NVML per UUID.

| Ripetizione | Inferenze | Durata (s) | Energia GPU (J) | J/image | Potenza media (W) |
|---|---:|---:|---:|---:|---:|
| 1 | 8.653 | 30,007110 | 2.051,815 | 0,237121807 | 68,377628 |
| 2 | 8.525 | 30,005721 | 2.047,060 | 0,240124340 | 68,222323 |
| 3 | 8.449 | 30,008467 | 2.035,607 | 0,240928749 | 67,834423 |

Aggregato originale: **0,23937573652788074 J/image**, potenza media
**68,14478564553075 W**, deviazione standard fra le medie delle finestre
**0,0020064512516075076 J/image**. Totali: 25.627 inferenze,
90,02129718199984 s e 6.134,482 J. Delta dei contatori, rapporti per finestra,
aggregati ponderati, FPS e incrementi di memoria controllati contro il JSON.
Le finestre energetiche sono separate da quelle della latenza; la deviazione
standard descrive le medie delle finestre. Si mantiene lo scope GPU completo
già documentato per TG-512, con idle incluso e CPU/loader/trasferimenti esclusi.

[Benchmark completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_384_layer2_sigma1/efficiency_20261007_093711/inference_benchmark_batch1.json),
[CSV derivato dal report](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_384_layer2_sigma1/efficiency_20261007_093711/inference_benchmark_batch1.csv),
[energia con contatori NVML](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_384_layer2_sigma1/efficiency_20261007_093711/inference_energy_batch1.json),
[console originale, copia identica](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_384_layer2_sigma1/efficiency_20261007_093711/user_console_output.txt),
[provenienza](../../outputs/anomaly_detection/local_registry_imports/2026-10-07/tinyglass_384_layer2_sigma1/efficiency_20261007_093711/import_provenance.json).

La registrazione conserva i precedenti artefatti TG-512. Il JSON console
precede il salvataggio finale e non contiene `benchmark_run_id`; il nome
della cartella locale deriva dal timestamp fornito. I confronti riguardano
le configurazioni complete, che differiscono in input, griglia e smoothing.

### Provenienza EfficientAD-S

Benchmark creato il `2026-10-06T15:07:10.382107+00:00` sul checkpoint
`efficientad_s_A_overhead_crop720_global_fixed70k_balanced__seed42__20260901_120051`,
Drive file ID `1LTxIQmEovzJTnosP6Plwrobvh7qBpmr5`.
Configurazione effettiva: input 256x256, 384 canali, posa A, crop 720,
calibrazione globale e score `global_max`. Inferenza FP32 senza autocast;
il default legacy `mixed_precision=True` e relativo al costruttore per il
training e non descrive la precisione di inferenza. Nessun training,
ricalibrazione o nuovo test di accuratezza e stato eseguito dal benchmark.

[Report originale completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/efficientad_s_pose_a/inference_benchmark_batch1.json):
tutte le distribuzioni di latenza, throughput, parametri/buffer/checkpoint,
memoria CUDA totale/incrementale, configurazioni salvata/ripristinata,
statistiche di calibrazione, preprocessing e metadati hardware/software.

[Riepilogo energetico fornito in chat](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/efficientad_s_pose_a/inference_energy_summary.json):
stato `measured`, backend `nvml_total_energy_counter`,
**4,031691 J/image**, potenza media
**66,645356 W**, deviazione standard tra le ripetizioni
**0,023861 J/image**, nessun warning riportato.

L'allegato originale contiene ancora `energy_joules_per_image: null`;
si conserva invariato e si collega il riepilogo successivo anziche riscriverlo.
Manca ancora il JSON completo delle finestre energetiche, con numero di
inferenze, tempi e contatori NVML per ripetizione. Il valore aggregato e
registrato come comunicato dall'utente, senza inventare queste diagnostiche.
La misura riguarda l'intera GPU nella fase predict-only, non CPU o pipeline
completa, e non sottrae il consumo a riposo.

### Provenienza SuperSimpleNet

Benchmark creato il `2026-10-06T20:05:36.588973+00:00`; fase energetica
completata il `2026-10-06T20:07:11.753252+00:00`. Checkpoint selezionato:
`supersimplenet_A_overhead_posecrop_input256_wheelmask_perlin02_score1pct_earlystop__seed42__20260905_093537`,
Drive file ID `1uAdzWRKpmQZnjECnAuEXD08CtDY8xfhv`, cartella
`1XPHgLw2jVMiRJRtwouqtIWFrQeEHDBeA`.
La configurazione salvata coincide con quella di inferenza: Wide-ResNet50-2,
layer2/layer3, input 256x256, posa A, crop 720, Perlin threshold 0,2,
score media del top 1% e sigma 4. Nonostante `wheelmask` nel nome storico,
`restrict_synthetic_anomalies_to_target_mask=False`: si tratta della variante
unrestricted selezionata, non di quella con anomalie ristrette alla ruota.
Manifest coincidente con la sorgente; validation con 296 clean e 96 hole.

[Benchmark completo](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/supersimplenet_pose_a/efficiency_20261006_200536/inference_benchmark_batch1.json),
[energia con finestre e contatori NVML](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/supersimplenet_pose_a/efficiency_20261006_200536/inference_energy_batch1.json),
[console originale](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/supersimplenet_pose_a/efficiency_20261006_200536/user_console_output.txt),
[provenienza dell'import](../../outputs/anomaly_detection/local_registry_imports/2026-10-06/supersimplenet_pose_a/efficiency_20261006_200536/import_provenance.json).
Nessun training, ricalibrazione, selezione dello scoring/sigma o nuovo test di
accuratezza: le metriche gia registrate restano associate al checkpoint.
Inferenza FP32 senza autocast; il forward canonico completo, inclusa la
decision branch, resta eseguito anche con scoring top-1%.
I precedenti benchmark di latenza restano conservati come misure separate.

Predict-only **27,5438 ms**, p95 **31,8189 ms**, **36,3058 immagini/s**;
H2D + predict **26,3263 ms**, end-to-end **78,9500 ms**. Anche qui le
finestre sono separate: non si ricava il tempo di trasferimento sottraendo
le due medie. Picco CUDA allocato **219,6885 MiB**, gia comprensivo del
modello e degli altri tensori vivi; nessuna memory bank. L'incremento
predict-only di 80 MiB non sostituisce il picco totale nel confronto.
Il conteggio dei parametri con `requires_grad` non implica training nel
benchmark, eseguito senza gradienti.

Energia misurata tramite `nvml_total_energy_counter`, warm-up effettivo
5,0241 s e tre finestre complete da circa 30 s, senza warning:

| Ripetizione | Inferenze | Durata (s) | Energia GPU (J) | J/image | Potenza media (W) |
|---|---:|---:|---:|---:|---:|
| 1 | 1.060 | 30,0294 | 1.982,616 | 1,870392 | 66,022591 |
| 2 | 1.008 | 30,0298 | 1.956,836 | 1,941306 | 65,163123 |
| 3 | 953 | 30,0112 | 1.917,864 | 2,012449 | 63,904976 |

Aggregato originale: **1,9388666005 J/image**, potenza media
**65,0304579232 W**, deviazione standard tra le medie delle finestre
**0,0710283588 J/image**. Totale: 3.021 inferenze, 90,070348 s e
5.857,316 J. Delta dei contatori in mJ, somme, aggregati ponderati e
dispersione verificati contro il report. La deviazione standard non e
quella dell'energia delle singole inferenze. Le finestre successive
mostrano meno inferenze e piu J/image; senza telemetria di clock o
temperatura non si attribuisce questa variazione a una causa specifica.

Scope: input residente, predict-only, intera GPU; CPU, loader e trasferimento
esclusi, nessuna sottrazione dell'idle o attribuzione al singolo processo.
CUDA/NVML associati per UUID; un solo processo compute rilevato e power
limit 70 W. Il JSON console non contiene `benchmark_run_id`: non e stato
inventato un identificativo assente.
