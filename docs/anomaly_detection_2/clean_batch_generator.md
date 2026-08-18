# Clean Batch Generator v1

Stato: implementato e verificato con smoke batch clean corrente da 24
campioni, incluso il gate `terrain_contact`. Questa milestone costruisce
l'infrastruttura; non genera il dataset finale e non introduce anomalie.

## Contratto

La configurazione operativa e `configs/blender/clean_batch.json`; pesi, master
seed e jitter restano definiti esclusivamente in
`configs/blender/domain_randomization.json`. La sorgente e il file immutabile
`outputs/anomaly_detection_2/pose_sampling/wheel_roll_pose_sampling.blend`, con
SHA-256 `9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`.

Il runner materializza prima un `plan.jsonl` canonico, poi usa un processo
Blender persistente per chunk. Camera, materiali per-ruota, controlli dust/wear
e shader AOV vengono predisposti una sola volta. Ogni campione ripristina lo
stato baseline, applica le scelte gia risolte e usa al massimo 64 retry che
possono cambiare soltanto il jitter camera.

Il post-processing PNG e sovrapposto al render successivo tramite un singolo
worker CPU e una coda limitata a quattro campioni. Il worker non usa mai `bpy`:
esegue soltanto split RGB/mask, compressione, verifica, hash e commit atomico.
Il benchmark dedicato e documentato in
[clean_performance_benchmark.md](clean_performance_benchmark.md).

Il rover viene traslato temporaneamente in XY per portare la ruota target
sull'anchor approvato di `wheel_middle_left`; terreno e georeferenziazione
restano fissi. La traslazione e registrata nel manifest.

## Pass e output

Ogni campione richiede una sola chiamata a render Eevee e produce:

- RGB PNG 8 bit, `1200x900`;
- maschera binaria `0/255` della ruota target, con occlusioni coerenti con RGB.

Blender 5.2 non espone Object Index nel Render Layers della scena. Il runtime
usa quindi uno shader AOV e lo trasporta temporaneamente nell'alpha dello
stesso PNG. Per evitare fringe sulle transizioni antialias, l'alpha e codificato
nell'intervallo 0,5--1 e l'RGB e premoltiplicato in modo coerente; il decoder
stdlib separa poi RGB e silhouette binaria senza un secondo render.

La directory di un run contiene `run.json`, `plan.jsonl`, `manifest.jsonl`,
`matrix_audit.json`, `rgb/`, `target_wheel_mask/` e `_staging/`. I path del
manifest sono relativi. Gli artefatti vengono verificati e mossi atomicamente
prima dell'append con `fsync`; un file non registrato non e completo.

## Uso

```powershell
python scripts/host/run_clean_batch.py `
  --config configs/blender/clean_batch.json `
  --run-id smoke_v2_terrain_contact_final `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe'
```

Resume fail-closed:

```powershell
python scripts/host/run_clean_batch.py `
  --config configs/blender/clean_batch.json `
  --run-id smoke_v2_terrain_contact_final --resume `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe'
```

Validator indipendente:

```powershell
python scripts/host/validate_clean_batch.py `
  --run-dir outputs/anomaly_detection_2/clean_batch/smoke_v2_terrain_contact_final
```

Probe di drift A--B--A:

```powershell
python scripts/host/run_clean_batch_drift_probe.py `
  --config configs/blender/clean_batch.json `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe' `
  --run-dir outputs/anomaly_detection_2/clean_batch/drift_probe_release
```

## Verifica eseguita

Lo smoke corrente `smoke_v2_terrain_contact_final`, sugli indici
`dr_000078`--`dr_000101`, ha prodotto 24 campioni validi con tutte le
sei ruote, quattro pose, due luci, tre livelli di usura e otto fasi di roll.
La matrice `6 ruote x 4 pose` e stata accettata `24/24`. Nel processo di render
`source_open_count=1` e `render_call_count=24`; i conteggi dei datablock sono
rimasti stabili. Tutti i 24 manifest record hanno `terrain_contact=true`,
`terrain_contact_details.ok=true`, framing e copertura terreno validi. Il
validator indipendente riporta 24 RGB e 24 target-wheel mask valide.

Il precedente `smoke_v1_contract_final` resta soltanto storico: i raster non
erano necessariamente errati, ma i suoi metadati precedono il gate di contatto
e vengono correttamente rifiutati dal validator corrente.

Il probe A--B--A ha restituito hash RGB e mask identici per i due A e metadati
identici per seed, sampling, trasformazioni, geometria e gate. Il resume
completo non ha modificato hash o timestamp dei 48 artefatti. Il checksum della
sorgente e rimasto invariato prima e dopo entrambi i test.

## Limiti v1

Anomaly injection/mask, depth, normal, split dataset, export Anomalib/MVTec,
parallelismo e batch da 10.000 immagini restano intenzionalmente esclusi.
La riproducibilita forte copre piano e metadati; l'identita pixel e garantita
solo a parita di Blender, motore, device e driver.
