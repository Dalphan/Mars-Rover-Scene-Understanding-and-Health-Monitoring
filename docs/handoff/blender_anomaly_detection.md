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
7. usa una shell parametrica manifold quando la pelle originale fallisce il
   repair gate;
8. applica Boolean Difference `EXACT`, crea un proxy di intersezione per la
   maschera e salva una coppia controfattuale;
9. genera report JSON/Markdown, due `.blend`, render e log.

Ultimo run reale formalmente valido:

- strategia: `hybrid_parametric_shell`;
- pelle finale senza boundary o multi-face edge;
- Boolean concluso e pelle perforata manifold;
- mask/wheel ratio: circa `0.005656`;
- differenza fuori dalla maschera dilatata: `0`;
- validator host: 14 file controllati, zero errori;
- GLB originale invariato.

Gli output erano locali sotto
`outputs/wheel_preparation/curiosity_middle_right/` e non sono versionati.

## Limite attuale: fase 2 non è ancora conclusa visivamente

Il passaggio automatico dei gate non equivale ancora a un risultato utilizzabile
per il dataset.

Il controllo visivo dell'ultimo run ha mostrato:

- la candidata grezza è molto frammentata e presenta grandi pannelli separati;
- alcune facce della pelle originale restano in `Wheel_Details` davanti alla
  shell ricostruita;
- la shell derivata risulta troppo chiara e ingloba visivamente parte dei
  rilievi;
- `normal.png` e `anomaly.png` possono apparire quasi identiche anche con una
  maschera non vuota;
- il validator controlla la differenza fuori dalla maschera, ma non impone
  ancora una differenza RGB minima dentro la maschera.

Pertanto non usare ancora `wheel_canonical.blend` per generare un dataset e non
iniziare i grousers rotti.

## Prossimi passi ordinati

### 1. Correggere la separazione pelle/dettagli

In `classify_skin_faces` il criterio per componente va integrato con una
selezione per faccia:

- calcolare per ogni faccia raggio cilindrico, allineamento della normale e
  intervallo radiale dei vertici;
- costruire `skin_faces` come unione delle componenti di pelle e delle facce
  nella banda del raggio base;
- rimuovere dalla copia `Wheel_Details` tutte le facce della pelle base, anche
  quando condividono una componente con un grouser;
- mantenere hub/razze e le superfici dei grousers sopra la banda;
- aggiungere al report conteggi distinti per selezione a componente e a faccia.

Verificare con render isolati `Wheel_Skin`, `Wheel_Details` e composito.

### 2. Migliorare la shell ibrida

- Ridurre luminosità/base color del materiale derivato; non collegare la texture
  atlas NASA alle UV cilindriche, perché mostra parti non correlate del rover.
- Fit di raggio e limiti assiali usando solo le facce radiali della pelle, non
  tutte le facce delle componenti selezionate.
- Conservare nel report materiale sorgente, metodo di derivazione, raggio,
  larghezza e spessore.
- Controllare che i grousers emergano sopra la shell senza z-fighting.

### 3. Rendere il gate controfattuale significativo

Aggiungere alla configurazione e al validator:

- `minimum_inside_change_ratio`;
- intensità media/minima della differenza RGB dentro la maschera;
- intersezione fra maschera e differenza RGB;
- fallimento se `normal.png` e `anomaly.png` sono identiche o il foro non è
  leggibile;
- opzionalmente un controllo sul contrasto del bordo del foro.

La maschera deve restare tra 0,2% e 5% della ruota visibile, ma deve anche
corrispondere a un cambiamento RGB effettivo.

### 4. Completare le verifiche previste dalla milestone

- Aggiungere una riapertura headless di `wheel_canonical.blend` e
  `perforation_probe.blend`.
- Verificare scene `Normal` e `Perforation`, camera/luci identiche e risorse
  disponibili.
- Aggiungere test unitari per i nuovi gate interni alla maschera.
- Rieseguire l'intera suite e il GLB reale; ispezionare almeno viste outboard,
  tread, oblique, normal, anomaly, mask e difference.

La fase 2 è conclusa solo quando gate automatici e QA visivo sono entrambi
positivi.

### 5. Dopo la fase 2

1. clustering cilindrico dei singoli grousers e contact sheet di verifica;
2. primo grouser rimosso/rotto in coppia controfattuale;
3. scena canonica minima con piccolo contesto rover e terreno;
4. camera/luce/materiali parametrizzati e domain randomization plausibile;
5. generazione RGB, wheel mask, anomaly mask e metadati per coppia;
6. QA automatico, manifest, split per seed/geometria e export MVTec/Anomalib;
7. pilot 500–1.000 coppie;
8. solo dopo: PatchCore e modelli addestrati con anomalie sintetiche;
9. valutazione finale su un piccolo campione MAHLI reale.

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
  --audit-report 'outputs\blender_audit\curiosity_static\reports\audit.json' `
  --candidate-id wheel_candidate_05 `
  --output-dir 'outputs\wheel_preparation\curiosity_middle_right'
```

```powershell
python scripts/host/validate_wheel_preparation.py `
  --output-dir 'outputs\wheel_preparation\curiosity_middle_right'
```

Non usare `--skip-validation` per un run candidato alla consegna.
