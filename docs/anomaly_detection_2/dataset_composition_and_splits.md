# Composizione e split del dataset v1

Stato: generazione bulk e assemblaggio finale completati. Il dataset canonico
contiene 10.000 immagini ed e descritto in
[final_dataset_assembly.md](final_dataset_assembly.md).

## Composizione fissata

Il dataset finale contiene 10.000 immagini complessive. L'unita atomica del
pool e un clean standalone oppure una coppia controfattuale clean/hole.

| Split | Clean standalone | Coppie | Immagini clean | Immagini hole | Totale |
|---|---:|---:|---:|---:|---:|
| train | 7.000 | 0 | 7.000 | 0 | 7.000 |
| validation | 500 | 250 | 750 | 250 | 1.000 |
| test | 0 | 1.000 | 1.000 | 1.000 | 2.000 |
| totale | 7.500 | 1.250 | 8.750 | 1.250 | 10.000 |

Il train e rigorosamente clean-only. Validation e test conservano il clean
controfattuale di ogni anomalia. Una coppia e indivisibile e il suo
`pair_lock_id` non puo comparire in split diversi.

## Distribuzioni produttive

- foro: 50% small, 40% medium, 10% large;
- superficie: 80% tread, 20% shoulder; large soltanto tread;
- camera: A 40%, C1 27,5%, C2 27,5%, D 5%;
- luce: dusty 75%, clear 25%;
- usura: current 20%, light 45%, evident 35%;
- ruota uniforme sulle sei ruote e roll uniforme sulle otto fasi.

Il contratto autorevole e `configs/blender/dataset_composition_v1.json`. I JSON
del pilot non vengono modificati: i loro hash fanno parte dei run canonici e
cambiarli distruggerebbe la riproducibilita storica. Il futuro orchestratore
bulk deve applicare questi pesi sopra i sampler congelati.

## Pool e vista ordinata

```text
outputs/anomaly_detection_2/datasets/<dataset_id>/
├── pool/
│   └── manifest.jsonl
├── images/<split>/<condition>/
├── masks/target_wheel/<split>/<condition>/
├── masks/anomaly/<split>/hole/
├── splits/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── dataset.json
```

Il manifest del pool usa una riga per unita. Un clean standalone ha un membro
`clean`; una coppia ha membri ordinati `clean`, `hole`, un unico
`pair_lock_id` e i campi semantici necessari alla stratificazione. La vista
ordinata sotto `images/` e `masks/` usa hardlink verso i run sorgente: e quindi
comoda da consumare ma non duplica i raster sul disco.

## Quote minime di valutazione

Le percentuali produttive da sole lascerebbero pochi esempi nella posa D e
nelle intersezioni con fori large. Validation e test impongono quindi quote
minime su ogni cella valida di ruota × camera × severita, ruota × camera ×
settore e ruota × camera × superficie. Il minimo e uno per cella in validation
e due in test. Il builder fallisce se il pool non contiene supporto sufficiente:
non sostituisce categorie e non separa coppie.

## Split materializzati

Gli split sono stati assegnati nel piano pre-render
[bulk_plan_v1.md](bulk_plan_v1.md). Il builder finale deve usare quegli indici
e verificarne gli hash; non deve effettuare una nuova selezione che potrebbe
cambiare le quote o separare il piano dagli artefatti prodotti.

L'assemblaggio conclusivo e stato eseguito con:

```powershell
python scripts/host/assemble_final_dataset.py `
  --config configs/blender/final_dataset_assembly_v1.json
```

Il risultato e fail-closed: non sovrascrive una vista gia pubblicata e verifica
hash, piano, quote, manifest e hardlink prima del commit.

## Rischi residui

- Le metriche paired vanno accompagnate da analisi per sottogruppo: le due
  immagini della coppia sono intenzionalmente correlate.
- L'export fisico MVTec/Anomalib resta derivato dagli indici e separato dal
  pool canonico.
- I raster della vista finale non devono essere modificati in place, perche gli
  hardlink condividono il contenuto con i run sorgente immutabili.
