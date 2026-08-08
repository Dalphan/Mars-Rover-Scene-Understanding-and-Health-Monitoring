# Handoff Codex — pipeline Blender per anomaly detection delle ruote

## Obiettivo del progetto

Generare un dataset sintetico RGB per il visual self-health monitoring delle
ruote di Curiosity. L'MVP riguarda perforazioni/strappi della pelle e grousers
rotti o mancanti. Polvere, sabbia e ombre saranno hard negative, non anomalie.

La pipeline resta divisa in due ambienti:

- Python host: lancio job, configurazioni, validazione, indicizzazione ed export;
- Python di Blender: importazione GLB, geometria, scene e rendering tramite
  `bpy`/`bmesh`.

Non inserire dipendenze ML nel Python di Blender.

## Ambiente e asset

- Versione verificata: Blender 5.2.0 LTS.
- Rendering iniziale: Eevee, 800x600, aspect ratio 4:3.
- GPU dell'ultimo run: Intel Iris Xe via OpenGL; la pipeline non richiede una
  GPU dedicata.
- Asset locale non versionato: `24584_Curiosity_static.glb`.
- SHA-256 atteso:
  `86a8ee6d39fb1711ae549ba99951d2f52b44a7b8367ce4814cc81aec60e94b31`.
- Non scaricare o sostituire automaticamente il GLB con glTF. L'audit non ha
  trovato texture mancanti o perdita di dati che giustifichi il glTF.

### Blender 5.1 sull'altro PC

Usare Blender 5.2.0 è la scelta raccomandata. Il preflight di fase 2 confronta
la coppia major/minor del Blender corrente con quella registrata dal report di
audit: il report 5.2 esistente viene quindi rifiutato correttamente da Blender
5.1.

Se non è possibile installare 5.2, Blender 5.1 può essere provato, ma occorre:

1. rieseguire l'audit reale con Blender 5.1 in una nuova directory;
2. validare quel nuovo audit;
3. passare il nuovo `audit.json` alla preparazione della ruota;
4. rieseguire tutti i test e ispezionare i render, perché la compatibilità API
   5.1 non è stata ancora verificata.

Non indebolire il controllo di versione per riutilizzare il report 5.2.

## Stato implementato

### Fase 1: audit GLB

Sono implementati:

- `scripts/blender/audit_asset.py`;
- launcher e validator host in `scripts/host/`;
- configurazione `configs/blender/audit.json`;
- logica pura e test in `src/blender_audit/` e `tests/`.

L'ultimo audit reale ha prodotto:

- Blender 5.2.0, Eevee, 800x600;
- un solo oggetto mesh `MSL` con 63.516 vertici, 51.305 facce e 6.946
  componenti disconnesse;
- sei candidate ruota incorporate nella mesh grande;
- categoria B con confidenza 0,965;
- nessuna texture mancante e checksum sorgente invariato.

La candidata di fase 2 è `wheel_candidate_05`: 303 componenti, 2.458
vertici, 3.970 edge e 1.664 facce.

### Fase 2: ruota canonica e perforazione

Sono presenti:

- `scripts/blender/prepare_wheel.py`;
- `scripts/host/run_wheel_preparation.py` e
  `scripts/host/validate_wheel_preparation.py`;
- `scripts/blender/build_perforation_library.py` e il launcher/validator
  `scripts/host/run_perforation_library.py`;
- `configs/blender/wheel_preparation.json`;
- logica/validator in `src/wheel_preparation/`;
- otto test host focalizzati.

Il flusso implementato:

1. verifica hash, report, Blender major/minor e topologia importata;
2. ricostruisce lo stesso ordinamento deterministico delle componenti;
3. duplica l'intera mesh e separa la candidata via Edit Mode/BMesh, senza
   ricostruire l'asset con `from_pydata`;
4. preserva coordinate, materiali per faccia, UV e normali custom;
5. canonicalizza centro/orientamento, stimando correttamente l'asse ruota X
   dalla disposizione 2x3 delle sei candidate;
6. esegue una sweep merge-by-distance sulla pelle originale;
7. usa una shell parametrica manifold come supporto quando la pelle originale
   fallisce il repair gate, ma mantiene mesh, materiale e UV NASA originali
   come superficie visibile;
8. applica Boolean Difference `EXACT`, crea un proxy di intersezione per la
   maschera e salva una coppia controfattuale;
9. importa il terreno libero "Mars Terrain Model" di John Davies
   (CC-BY-SA-4.0), conserva l'intero rover nella scena e genera overview e
   close-up;
10. genera report JSON/Markdown, due `.blend`, render e log.

Ultimo run reale verificato il 2 agosto 2026:

- Blender 5.2.0 LTS, Eevee, AMD Radeon RX 6600;
- candidata invariata: `wheel_candidate_05`;
- strategia: `textured_parametric_shell`, con pelle NASA originale visibile;
- classificazione pelle a componente + faccia: 507 facce di pelle;
- filtro inviluppo dettagli: 1.157 -> 1.086 facce rotanti; le 71 facce dei
  sette componenti inboard sono conservate separatamente in
  `Wheel_Attachment`;
- raggio shell `0.2341126534`, spessore `0.00746689`;
- pelle normale e perforata entrambe closed manifold;
- mask/wheel ratio `0.0061530345`;
- cambiamento RGB dentro la maschera `0.9833770779`;
- differenza RGB media dentro la maschera `0.2638773263`;
- differenza fuori dalla maschera dilatata `0`;
- validator host: 26 file controllati, zero errori;
- riapertura headless valida di entrambi i `.blend`;
- checksum del GLB originale invariato.

Gli output locali sono sotto
`outputs/wheel_preparation/curiosity_middle_right/` e non sono versionati.

## Fase 2 conclusa

Il QA visivo del run del 1 agosto 2026 è positivo per la milestone della ruota
canonica e della prima perforazione:

- i grandi pannelli della pelle originale non restano più davanti alla shell;
- le componenti inboard individuate con contact sheet sono escluse dalla parte
  rotante ma conservate come attacco fisso originale; hub e razze rimangono;
- il materiale texture-free della shell è grigio scuro e non usa l'atlas NASA;
- `normal.png` e `anomaly.png` differiscono chiaramente nel foro;
- `anomaly_mask.png` coincide con il cambiamento RGB mostrato in
  `difference.png`;
- non sono presenti cambiamenti RGB misurabili fuori dalla maschera dilatata.

La pipeline genera inoltre render isolati `skin_only.png`, `details_only.png`,
`attachment_only.png`, `composite.png`, `assembly_with_attachment.png`,
`assembly_inboard.png`,
`normal_skin_only.png`, `anomaly_skin_only.png` e `details_at_hole.png` per
rendere ripetibile il QA visivo.

Il bug che rendeva sana e anomala identiche era l'orientamento inward delle
facce del cutter. Le calotte e le pareti sono ora orientate outward; il validator
di riapertura controlla anche che gli hash geometrici delle due pelli differiscano.

## Gate e verifiche aggiunti

- selezione cilindrica per faccia basata su raggio del centro, allineamento della
  normale e intervallo radiale dei vertici;
- fit di raggio e limiti assiali solo dalle facce radiali selezionate;
- report separato dei conteggi per selezione a componente e a faccia;
- filtro deterministico delle componenti oltre l'inviluppo assiale: le parti
  esterne alla geometria rotante sono preservate come `Wheel_Attachment`, con
  ruolo `fixed_inboard_wheel_attachment` e `rotates_with_wheel = false`;
- materiale ibrido con luminanza massima configurabile e atlas disconnesso;
- `minimum_inside_change_ratio` e `minimum_inside_mean_difference`;
- metriche di intersezione maschera/differenza, intensità interna e coppia
  identica/non identica;
- riapertura headless di `wheel_canonical.blend` e `perforation_probe.blend` con
  verifica di scene, oggetti, camera, luci, render settings e risorse;
- test host focalizzati per i nuovi classificatori e gate.

## Prossimi passi ordinati

1. clustering cilindrico dei singoli grousers e contact sheet di verifica;
2. prima rimozione/rottura di un grouser in coppia controfattuale, con gli stessi
   gate interni ed esterni alla maschera;
3. validare manualmente piu pose close-up nel contesto rover/terreno gia integrato;
4. camera, luce e materiali parametrizzati con domain randomization plausibile;
5. generazione RGB, wheel mask, anomaly mask e metadati per coppia;
6. QA automatico, manifest, split per seed/geometria ed export MVTec/Anomalib;
7. pilot da 500–1.000 coppie;
8. solo dopo: PatchCore e modelli addestrati con anomalie sintetiche;
9. valutazione finale su un piccolo campione MAHLI reale.

Non avviare ancora la generazione bulk: il prossimo gate manuale è la prima
coppia controfattuale con grouser rimosso/rotto.

## Simulatore rover - stato corrente: Milestone 1

Il 2 agosto 2026 il simulatore e stato riportato intenzionalmente alla sola
gerarchia statica della Milestone 1:

- `rover_simulator.blend` viene derivato da `perforation_probe.blend`, senza
  modificare il file sorgente;
- la scena persistente `Simulator` condivide il rover e la ruota sana con la
  coppia controfattuale, ma resta senza terreno in attesa del nuovo asset;
- `RoverRoot` controlla corpo, `Wheel_Attachment`, varianti sana/anomala e
  `WheelCamera`, mentre il terreno resta indipendente;
- quattro probe sono derivati deterministicamente dalle ruote anteriori e
  posteriori dei due lati dell'audit 2x3;
- `SimulatorCamera` fornisce la vista inseguitrice statica e `WheelCamera`
  mantiene il close-up di `wheel_candidate_05`;
- il derivato corregge `Wheel_Skin_Normal.hide_render`, rimasto attivo nel
  blend sorgente sulla copia di scena nonostante la geometria valida;
- i render diagnostici mostrano overview, impronta dei probe e close-up;
- la riapertura headless verifica che una traslazione temporanea di
  `RoverRoot` muova tutto il rover;
- digest delle pelli sana/anomala invariati e sorgente immutata;
- `Wheel_Attachment` resta esplicitamente figlio di `RoverRoot`, preservando la
  correzione del collegamento tra ruota e rover;
- mosaico, collider, raycast, solver di posa, normali di contatto e metadata
  delle Milestone 2/2.5 sono stati rimossi dal codice e dal `.blend`;
- `RoverRoot` e nuovamente alla trasformazione identita;
- i PNG prodotti in precedenza sono mantenuti soltanto come riferimenti visivi
  per la scelta del nuovo terreno e non descrivono lo stato corrente del blend.

Output locale:
`outputs/rover_simulator/curiosity_middle_right/diagnostics/rover_simulator.blend`.

Il prossimo passo e scegliere o costruire il nuovo terreno. Collisione e
risolutore di posa dovranno essere reintrodotti solo dopo la sua approvazione;
non iniziare ancora il controller WASD.

### Anteprima visiva del nuovo terreno (4 agosto 2026)

- importata soltanto la mesh `Plane` da `martian-terrain/source/Martian Terrain.zip`;
- rinominata `MartianTerrain` e collegata alla scena `Simulator`, senza camera
  o luce dell'asset sorgente;
- scalata a 50x50 m con rilievo verticale complessivo di 0,9 m;
- collegate `Diffuse.png`, `Normal.png` e `roughness.png`; displacement escluso;
- mantenute camera e luci M1; rover e `RoverRoot` non riposizionati;
- nessun collider, raycast o solver introdotto;
- i render `new_terrain_overview.png` e `new_terrain_oblique.png` mostrano un
  hotspot centrale e ombre quasi nere prodotte dalla combinazione fra luce M1
  e normal map. La correzione resta sospesa fino alla scelta dell'utente.
- dopo approvazione, la scena `Simulator` usa una luce `SUN` dedicata con
  energia 2, altezza apparente 35 gradi e angolo 4,5 gradi, oltre a un world
  ambientale debole con strength 0,08;
- `PAIR_RIG` e escluso soltanto da `Simulator`; `WheelCamera` viene ricollegata
  direttamente, mentre `Normal` e `Perforation` conservano le luci originali;
- i render `new_terrain_solar_overview.png` e
  `new_terrain_solar_oblique.png` confermano illuminazione uniforme e assenza
  dell'hotspot. Le chiazze nere persistono e non sono contenute nella diffuse:
  la forza della normal map resta il prossimo parametro da valutare, senza
  averlo ancora modificato.
- su conferma, l'energia del Sole e stata portata da 2 a 3,5;
- sono state renderizzate varianti zenitali e oblique con normal strength 0,30
  e 0,15. Entrambe eliminano quasi tutte le chiazze nere: 0,30 mantiene piu
  micro-rilievo, 0,15 risulta piu uniforme;
- durante il confronto il materiale era rimasto a strength 0,65, senza
  applicare automaticamente una delle due varianti;
- l'utente ha selezionato strength 0,30, ora salvato nel materiale;
- il rover e stato posizionato nella zona pianeggiante sinistra a
  `(-18, -4, 0,342345)`, yaw 45 gradi. L'audit temporaneo dei quattro probe
  misura dislivello 0,023902 m e pendenza media 0,530818 gradi;
- la posa e soltanto visuale: nessun collider o solver persistente e presente;
- prodotti `terrain_with_rover_overview.png`,
  `terrain_with_rover_close.png` e `terrain_with_rover_wheel_close.png`.
- applicato su conferma il preset `mars_clear_day_closeup`: una sola Sun a
  380 W/m2, angle 0,35 gradi, colore (1, 0,97, 0,92), elevazione 40 gradi,
  azimut 135 gradi, World 0,035, AgX Medium High Contrast ed esposizione 0 EV;
- `Simulator`, `Normal` e `Perforation` condividono lo stesso World e la stessa
  collezione luce; `PAIR_RIG` e escluso e `WheelCamera` resta disponibile;
- roughness terreno rimappata nell'intervallo 0,85-0,95 e normal mantenuta 0,30;
- QA visivo non superato: alla scala fotometrica/materiale corrente 380 W/m2 a
  0 EV brucia terreno, metalli e texture della ruota. Non modificare exposure o
  strength senza una nuova decisione dell'utente.
- provato il preset successivo autorizzato a 170 W/m2, esposizione -1 EV e
  World 0,03; ancora troppo chiaro nel terreno e al centro della ruota;
- applicato quindi il fallback autorizzato a 120 W/m2 e -1,5 EV. Emission
  Strength del terreno verificato e impostato a zero;
- il fallback migliora grousers e dettagli metallici, ma il terreno resta
  pallido e il centro della ruota ancora luminoso. Attendere approvazione prima
  di ulteriori variazioni.
- collegato lo stesso oggetto `MartianTerrain` a `Simulator`, `Normal` e
  `Perforation`, senza duplicare mesh o materiale;
- applicati i valori approvati: Sun 100 W/m2, angle 0,35 gradi, exposure -1,5
  EV, World 0,018, normal 0,25, roughness 0,90-0,95 e moltiplicatore Base Color
  0,625; emissione terreno zero;
- i close-up `shared_terrain_normal_wheel.png` e
  `shared_terrain_perforation_wheel.png` hanno ora lo stesso terreno sullo
  sfondo. Le ombre del rover risultano molto nette per l'angle 0,35 gradi;
- il validator impone terreno, World, exposure e collezione Sun identici nelle
  tre scene quando `terrain_shared_with_pair_scenes` e attivo.
- applicato il preset di prova successivo: Sun 60 W/m2, angle 0,35 gradi,
  exposure -1,5 EV, World 0,01, normal 0,25, roughness 0,90-0,95 e
  moltiplicatore Base Color 0,40; l'emissione del terreno resta zero;
- i render aggiornati sono `shared_terrain_rover_close.png`,
  `shared_terrain_normal_wheel.png` e
  `shared_terrain_perforation_wheel.png`. Il terreno e visibilmente piu scuro,
  mentre la ruota rimane leggibile; il preset e ancora in attesa della
  valutazione visiva dell'utente.
- sostituito il precedente moltiplicatore Base Color con la catena isolata nel
  solo materiale `MartianTerrain_Visual`: `Diffuse` -> Hue/Saturation
  (Hue 0,52, Saturation 1,55, Value 0,90) -> ColorRamp -> Principled Base
  Color. La ColorRamp usa, in sRGB, `#3D1408`, `#8F3515` e `#C7682D`;
  roughness 0,90-0,95 e normal 0,25 restano invariate;
- Sun 60 W/m2, angle 0,35 gradi, exposure -1,5 EV e World 0,01 restano
  invariati. I render mostrano un terreno piu rosso-aranciato senza modificare
  il materiale o l'esposizione del rover; attende valutazione visiva.
- su richiesta dell'utente e stato eseguito il revert al preset approvato nella
  precedente annotazione: `SimulatorSolarSun` caldo a energia 3,5, posizione
  `(-30,-40,35)` (elevazione 35 gradi), angolo 4,5 gradi e World strength
  0,08; esposizione ripristinata a 0 EV;
- il materiale `MartianTerrain_Visual` usa nuovamente le mappe Diffuse e
  Roughness originali direttamente nel Principled e normal map 0,30. La catena
  Hue/Saturation/ColorRamp non e piu collegata; posa rover, probe nascosti,
  `Wheel_Attachment` e assenza di collider/solver restano invariati;
- il preset non impone piu il contesto condiviso Simulator/Normal/Perforation:
  e un ritorno deliberato al preview di simulazione precedente. Render:
  `annotated_preset_rover_close.png`.
- su richiesta successiva il Sole del preset annotato e stato alzato a 70 gradi
  sopra l'orizzonte, mantenendo energia 3,5, angolo 4,5 gradi, azimut e tutti
  gli altri parametri invariati, per accorciare le ombre.
- corretta la rotazione del Sole dopo il cambio di elevazione: la direzione dei
  raggi e ora calcolata dalla nuova posizione, non da una matrice obsoleta.
  Verifica geometrica: prodotto scalare tra raggio e direzione verso il terreno
  pari a 1,0; il render aggiornato e `annotated_preset_rover_close.png`.
- completato il dettaglio locale A/B/C in un derivato separato,
  `rover_simulator_detailed.blend`: tre patch da 3x3 m, griglia a 2 cm,
  displacement deterministico multiscala, bump sfumato, contatto statico e
  rocce procedurali. I seed sono 4101, 4102 e 4103; i profili sono piatto,
  roccioso e lieve pendenza;
- implementato il materiale ibrido interno a Blender per i soli close-up:
  `Texture Coordinate (Generated)` -> `Mapping` -> due Noise multiscala per
  variazione cromatica e roughness, con il Noise fine collegato realmente al
  Bump insieme alla normal map originale. La roughness locale viene rimappata
  in `0,88-0,98`, senza modificare il materiale della ruota o la luce globale;
- le rocce delle patch usano copie dedicate del materiale del terreno con
  Noise/Voronoi per albedo, roughness e bump, più Subdivision Simple e Displace
  procedurale lieve (`0,006 m`). In questo modo la superficie resta originale
  fuori dalle patch e il dettaglio ruvido è confinato alle aree campionate;
- Patch A e centrata sul contatto della ruota osservata a
  `(-17,9590,-3,9590)`, mentre B e C usano rispettivamente i centri
  `(0,10)` e `(8,-12)`. Tutte riusano UV e texture originali e sono condivise
  tra `Simulator`, `Normal` e `Perforation`;
- la posa rover viene trasferita staticamente al centro di ciascuna patch e la
  `WheelCamera` viene riallineata prima di ogni render. Collider, solver,
  controller WASD e deformazione dinamica restano disabilitati;
- render prodotti per ogni patch: overview, zenitale, ruota sana e ruota
  perforata (`patch_A/B/C_*.png`). Validator simulatore, riapertura della
  coppia perforata e test host focalizzati superati; il batch e in attesa di
  approvazione visiva finale.
- creato il backup reversibile
  `rover_simulator_detailed_backup_before_regolith_pilotA.blend` e una copia
  separata `rover_simulator_regolith_pilotA.blend`. Solo `TerrainPatch_A` e
  stata aggiornata con materiale macro/meso/micro, 233 frammenti Poisson in
  tre bande, 36 rocce in quattro famiglie e maschera polvere/impronta statica;
  `TerrainPatch_B/C` e la scena dettagliata precedente restano invariati;
- i render della variante usano il prefisso `regolith_pilotA_A_*.png`.
  Validator simulatore e riapertura della coppia perforata superati; la
  variante resta in attesa di approvazione visiva prima dell'estensione a B/C.
- su richiesta e stata creata una nuova copia completa
  `rover_simulator_regolith_all_patches.blend`, con backup
  `rover_simulator_regolith_all_patches_backup_before_extension.blend`.
  Le quattro modifiche sono ora applicate a A/B/C; le soglie minime delle
  rocce principali sono rispettivamente 10, 14 e 12 mm. I render di confronto
  usano il prefisso `regolith_all_A/B/C_{normal|perforation}.png`.
- la scena completa supera validator simulatore, riapertura della coppia e
  test host focalizzati; resta in attesa del feedback visivo prima di
  considerarla il nuovo riferimento.
- provata la variante di illuminazione invertita richiesta per la patch pilota:
  azimut 53,13 gradi, Sole a 70 gradi, energia 3,5 e angle 4,5 invariati.
  I validator passano, ma il close-up mostra il lato osservato della ruota piu
  scuro; la variante resta salvata per confronto e non e ancora approvata.

### Milestone 1 del piano di realismo: camera e contatto (5 agosto 2026)

- creata la copia di lavoro `rover_simulator_m1_camera_contact.blend` a partire
  da `rover_simulator_regolith_all_patches.blend`; il backup reversibile e
  `rover_simulator_m1_camera_contact_backup.blend`;
- il `RoverRoot` della copia viene abbassato soltanto quanto basta per lasciare
  4 mm tra il punto piu basso della ruota osservata e il terreno di
  `TerrainPatch_A` (superficie a `z=0,0775967 m`, fondo ruota a
  `z=0,0815967 m`); non e stato introdotto alcun collider, raycast persistente
  o solver;
- aggiunto `TerrainPatch_A_ContactBerm_M1`, una berm ellittica statica a 48
  segmenti, con materiale di polvere compatta, per rendere leggibile il punto
  di contatto nei futuri diagnostic render;
- `WheelCamera` mantiene la ruota e il terreno nella stessa inquadratura
  approvata: lente 50 mm, target spostato di `-0,08 m` sotto il centro ruota;
  la luce e il materiale del terreno non sono stati modificati;
- render verificati: `m1_approved_A_normal.png` e
  `m1_approved_A_perforation.png`; la differenza resta confinata alla ruota;
- validator di riapertura simulatore e coppia perforata superati, così come i
  test host focalizzati. Questa copia resta il backup funzionale della fase
  camera/contatto usato come base per la Milestone 2.

### Milestone 2 del piano di realismo: materiale regolith multiscala (5 agosto 2026)

- creata la copia `rover_simulator_m2_regolith.blend` dalla scena M1; il backup
  reversibile è `rover_simulator_m2_regolith_backup.blend`;
- applicato alle patch A/B/C un materiale interno a Blender che conserva le
  mappe Diffuse, Roughness e Normal originali e aggiunge variazioni separate
  macro, meso, micro e granulare;
- la nuova catena aggiunge albedo a basso contrasto, roughness granulare
  rimappata nell'intervallo `0,88-0,99` e un bump fine sovrapposto alla normal
  originale (`0,0012 m`, strength `0,34`);
- non sono state modificate `WheelCamera`, `SimulatorCamera`, la luce, il World,
  la posa del rover o la geometria della ruota; collisioni e solver restano
  disabilitati;
- render diagnostici prodotti: `m2_A_normal.png`, `m2_A_perforation.png` e
  `m2_A_zenith.png`. La coppia sana/anomala mantiene lo stesso contesto del
  terreno e differisce solo per la perforazione;
- validator di riapertura simulatore e coppia perforata superati. Questa copia
  resta il riferimento M2 usato come base per la Milestone 3; la valutazione
  visiva del materiale resta archiviata nel report M2.

### Milestone 3 del piano di realismo: libreria rocce (5 agosto 2026)

- creata la copia `rover_simulator_m3_rock_library.blend` dalla scena M2; il
  backup reversibile è `rover_simulator_m3_rock_library_backup.blend`;
- sostituite soltanto le mesh `TerrainPatch_A/B/C_Rocks`, mantenendo gli stessi
  punti Poisson, conteggi e seed di distribuzione della M2;
- introdotte quattro famiglie deterministiche: `angular`, `stratified`,
  `rounded` e `shard`, con anelli a 10 segmenti, chiusura watertight, bevel da
  `1,5 mm` e micro-displacement da `1,8 mm`;
- la libreria passa rispettivamente a 1.512, 2.688 e 2.016 vertici per le
  patch A/B/C, senza modificare la ruota o il materiale regolith M2;
- camera, luce, posa, terreno di base, collisioni e solver sono invariati;
- render prodotti: `m3_A_normal.png`, `m3_A_perforation.png` e
  `m3_A_zenith.png`. Validator simulatore, coppia perforata e test host
  focalizzati superati;
- la copia M3 resta il riferimento geometrico usato dalla Milestone 4; la sua
  libreria è conservata integralmente nel backup M3.

### Milestone 4 del piano di realismo: distribuzione clusterizzata (5 agosto 2026)

- creata la copia `rover_simulator_m4_clustered_distribution.blend` dalla
  scena M3; il backup reversibile è
  `rover_simulator_m4_clustered_distribution_backup.blend`;
- rocce e gravel sono stati ricollocati con campionamento clusterizzato
  deterministico, mantenendo libera l’ellisse di contatto della ruota;
- patch A/B/C usano rispettivamente 6, 8 e 6 cluster di rocce, con una quota
  di elementi dispersi (`22%`) per evitare una distribuzione artificiale;
- i conteggi restano invariati: 36/64/48 rocce e 233/284/233 isole gravel;
  l’allocazione interna dei cluster è bilanciata e registrata nel report;
- geometria M3, materiali M2, camera, luce, posa, ruota, collisioni e solver
  non sono stati modificati;
- render prodotti: `m4_A_normal.png`, `m4_A_perforation.png` e
  `m4_A_zenith.png`. Validator simulatore, coppia perforata e test host
  focalizzati superati;
- la variante M4 è stata conservata come storico ma non è più la scena attiva:
  su richiesta è stato eseguito il revert alla distribuzione uniforme M3 prima
  del tuning M5.

### Milestone 5 del piano di realismo: tuning illuminazione e render (5 agosto 2026)

- registrato il revert a `rover_simulator_m3_rock_library.blend` tramite la
  copia `rover_simulator_reverted_to_m3.blend`; M4 resta disponibile soltanto
  come confronto reversibile;
- creata la scena attiva `rover_simulator_m5_lighting.blend`, con backup
  `rover_simulator_m5_lighting_backup_m3.blend`;
- mantenuta una sola Sun attiva (`SimulatorSolarSun`): energia `3,5`,
  elevazione `70°`, angolo portato a `6°` per ammorbidire le ombre e colore
  solare leggermente caldo `(1,00, 0,94, 0,88)`;
- World uniformato a grigio-arancio desaturato, strength `0,09`; AgX Medium
  High Contrast, esposizione `0 EV`, Cycles `64` samples con denoising e
  risoluzione `800x600`;
- terreno, libreria rocce M3, distribuzione uniforme M3, camera, posa e
  geometria Normal/Perforation non sono stati modificati;
- render prodotti: `m5_A_normal.png`, `m5_A_perforation.png` e
  `m5_A_zenith.png`. Validator simulatore, coppia perforata e test host
  focalizzati superati;
### Milestone 6 del piano di realismo: feather delle transizioni (5 agosto 2026)

- derivata dalla scena M5 la build attiva
  `rover_simulator_m6_feather.blend`, con copia reversibile
  `rover_simulator_m6_feather_backup.blend`;
- applicato un attributo `PatchFeather` di `0,35 m` a tutte le patch A/B/C;
  la fascia raccorda colore diffuso originale, roughness e bump, lasciando
  invariata la texture del terreno fuori dalla patch;
- la densità della ghiaia viene sfumata verso il bordo (floor `0,20`, scala
  minima `0,55`) e il displacement viene interpolato verso la superficie;
  quando il terreno importato presenta un foro sorgente viene usato un piano
  locale fit per mantenere la continuità della superficie;
- la geometria Normal/Perforation, la camera dataset e l'illuminazione restano
  invariate; `collision_enabled` resta `false`;
- render di controllo prodotti: `m6_A_normal.png`, `m6_A_perforation.png` e
  `m6_A_zenith.png`. I validator di riapertura simulatore/coppia e i test host
  focalizzati sono passati. L'eventuale micro-seam del foro sorgente non è
  bloccante per questa iterazione, secondo la decisione dell'utente di non
  procedere con ulteriore tuning dell'overlap;
- report: `terrain_detail_m6_feather.json`,
  `m6_feather_simulator_reopen.json`, `m6_feather_pair_reopen.json`.

### Libreria M6 della perforazione irregolare (6 agosto 2026)

- dopo l'approvazione del pilot `perforation_02_axial_jagged_slit_medium`, il
  builder produce tutte le 12 combinazioni di orientamento, famiglia e taglia;
- il builder apre `rover_simulator_m6_feather.blend` in sola lettura e salva la
  copia derivata `rover_simulator_m6_irregular_perforation_library.blend`;
- la nuova pelle anomala deriva sempre da `Wheel_Skin_Normal`: non riusa
  `Wheel_Skin_Anomaly`, quindi il vecchio foro circolare non viene ereditato;
- parenting, `matrix_parent_inverse` e `matrix_local` sono preservati. I dettagli
  della ruota non sono duplicati né riposizionati; questo elimina le grandi
  strisce nere prodotte dai pilot precedenti;
- la posizione viene cercata dalla camera M6, rifiutando viste radenti, pelle
  opposta, aperture native e occlusioni vicine. Tutte le forme condividono un
  centro nella metà bassa della ruota, separato dai fori rettangolari originali;
- il ray-cast esclude esplicitamente la copia anomala corrente: in precedenza il
  nome variante non conteneva `Wheel_Skin` e la pelle si auto-classificava
  erroneamente come foreground;
- un canale centrale di 4 mm, contenuto nel contorno esterno, garantisce la linea
  di vista della camera attraverso lo spessore. Uno spot fisico condiviso dalla
  coppia, 100 W e 4 gradi, rende leggibile il terreno in ombra senza hotspot
  esterno o materiale nero fittizio;
- il cutter normalizza il winding dei contorni: lo scambio di assi dei profili
  circonferenziali non inverte più il solido Boolean;
- Boolean `EXACT` riuscito sulla pelle visiva e sulla shell strutturale per
  12/12 varianti; ogni shell resta closed manifold e tutti i 7 o 8 campioni
  radiali attraversano l'apertura. Inoltre 84/84 raggi camera attraversano il
  nucleo e terminano su `TerrainPatch_A`;
- prodotti `normal.png`, `anomaly.png`, `damage_mask.png`, `effect_mask.png`,
  `difference.png`, report JSON, contact sheet generale e ravvicinata e il
  `.blend` derivato. Tutti i gate automatici e la riapertura Blender 5.2 passano;
  resta la selezione visiva delle forme da conservare prima del bulk;
- gli undici output sperimentali falliti e il validator CLI duplicato sono stati
  rimossi. Il launcher dedicato usa direttamente il validator testabile in
  `src/wheel_preparation/validation.py`.

## Comandi di riproduzione con Blender 5.2

Collocare il GLB localmente senza committarlo. Dalla root della repo:

```powershell
python scripts/host/run_blender_audit.py `
  --blender 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --asset '<path>\24584_Curiosity_static.glb' `
  --output-dir 'outputs\blender_audit\curiosity_static'
```

```powershell
python scripts/host/run_wheel_preparation.py `
  --blender 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --asset '<path>\24584_Curiosity_static.glb' `
  --terrain '<path>\mars_terrain_model\scene.gltf' `
  --audit-report 'outputs\blender_audit\curiosity_static\reports\audit.json' `
  --candidate-id wheel_candidate_05 `
  --output-dir 'outputs\wheel_preparation\curiosity_middle_right'
```

```powershell
python scripts/host/validate_wheel_preparation.py `
  --output-dir 'outputs\wheel_preparation\curiosity_middle_right'
```

```powershell
python scripts/host/run_perforation_library.py `
  --blender 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --source-blend 'outputs\wheel_preparation\curiosity_middle_right\diagnostics\perforation_probe.blend' `
  --library-config 'configs\blender\perforation_library.json' `
  --output-dir 'outputs\wheel_preparation\curiosity_middle_right\perforation_library'
```

Il launcher del pilot esegue direttamente il validator host; non esiste un
secondo entrypoint CLI di sola validazione.

```powershell
python scripts/host/build_rover_simulator.py `
  --blender 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --source-blend 'outputs\wheel_preparation\curiosity_middle_right\diagnostics\perforation_probe.blend' `
  --audit-report 'outputs\blender_audit\curiosity_static\reports\audit.json' `
  --preparation-report 'outputs\wheel_preparation\curiosity_middle_right\reports\preparation.json' `
  --output-dir 'outputs\rover_simulator\curiosity_middle_right'
```

Non usare `--skip-validation` per un run candidato alla consegna.
