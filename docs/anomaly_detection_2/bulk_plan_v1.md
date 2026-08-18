# Piano bulk deterministico v1 — 10.000 immagini

## Stato

Piano completo materializzato e validato. La prima wave bulk da 1.000 clean
train e stata avviata ed e documentata in [bulk_wave_001.md](bulk_wave_001.md).
Il piano include i 200 raster RGB del pre-bulk e assegna in anticipo ogni
unita a train, validation o test.

Output autorevole:
`outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000/planning/bulk_v1`.

- `plan.json`: riepilogo, quote, hash e conteggi;
- `units.jsonl`: 8.750 unita atomiche con indice globale, sampling, split e
  stato `existing_prebulk` oppure `pending`;
- `splits/train.jsonl`, `validation.jsonl`, `test.jsonl`: assegnazione
  immutabile pre-render;
- `configs/blender/bulk_remaining_clean.json`: 7.350 clean ancora da rendere;
- `configs/blender/bulk_remaining_anomaly.json`: 1.225 coppie ancora da
  rendere, con categorie semantiche del foro materializzate esplicitamente.

Il piano e stato rigenerato due volte: JSONL, split e config hanno mantenuto
gli stessi SHA-256 byte per byte. `units.jsonl` ha SHA-256
`a03076f853d3634293758538ca0faa8ae576ef8452afe97ba1597a1c3609e192`.

## Composizione e riuso del pre-bulk

| Stato | Clean standalone | Coppie | Immagini |
|---|---:|---:|---:|
| gia prodotte | 150 | 25 | 200 |
| ancora da produrre | 7.350 | 1.225 | 9.800 |
| totale | 7.500 | 1.250 | 10.000 |

Le 150 unita clean del pre-bulk sono bloccate nel train. Le 25 coppie del
pre-bulk sono bloccate nel test. Non vengono ricodificate, copiate o
rigenerate. Tutti e quattro i gruppi di indici — clean pre-bulk, pair
pre-bulk, clean rimanenti e pair rimanenti — sono disgiunti.

Gli split pianificati sono esattamente:

- train: 7.000 clean standalone;
- validation: 500 clean standalone e 250 coppie;
- test: 1.000 coppie.

## Quote globali esatte

Le distribuzioni di dominio sono contate per immagine; ogni coppia contribuisce
due volte perche clean e hole condividono lo stesso stato.

- camera A/C1/C2/D: `4000/2750/2750/500`;
- luce dusty/clear: `7500/2500`;
- usura current/light/evident: `2000/4500/3500`;
- roll: `1250` immagini per ciascuna delle otto fasi;
- ruote: quattro ruote con `1667` immagini e due con `1666`, lo scarto minimo
  matematicamente possibile per 10.000 immagini divise su sei ruote.

Sulle 1.250 coppie:

- severita small/medium/large: `625/500/125`;
- superficie tread/shoulder: `1000/250`;
- settore leading/upper/trailing: `417/417/416`;
- profilo jagged/branched/peeled: `563/500/187`;
- tutti i large restano sul battistrada.

## Quote minime di evaluation

Il planner non si limita ai margini. Sono materializzate tutte le celle valide:

| Dimensione | Celle | Minimo validation | Minimo test |
|---|---:|---:|---:|
| ruota × camera × severita | 72 | 1 | 2 |
| ruota × camera × settore | 72 | 1 | 2 |
| ruota × camera × superficie | 48 | 1 | 2 |

La combinazione rara `camera D × large`, che con sampling indipendente non
avrebbe supporto sufficiente garantito, viene quindi inclusa deliberatamente
per tutte le ruote. Le categorie semantiche del foro sono override di piano
espliciti e fanno parte del fingerprint del config paired; il renderer non puo
sostituirle silenziosamente.

## Contratto e rigenerazione

Il comando autorevole e:

```powershell
python scripts/host/plan_bulk_dataset.py `
  --output-dir outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000/planning/bulk_v1
```

Una directory esistente fallisce salvo `--overwrite` esplicito. Il planner:

1. legge i due manifest pre-bulk validati;
2. costruisce i target globali e per split;
3. riserva prima le celle rare richieste dalle quote;
4. cerca indici deterministici che realizzano esattamente ruota, camera, luce,
   usura e roll;
5. materializza severita, superficie, settore e famiglia del foro;
6. ricostruisce entrambi i piani tramite i contratti clean/paired reali;
7. fallisce se una quota, un conteggio o un vincolo large→tread non coincide.

## Riesame critico

- Il piano garantisce categorie e split, non il successo futuro dei gate
  Blender. Una coppia che fallisce deve restare fallita; non va sostituita con
  un indice di altra categoria. L'eventuale sostituzione richiedera una nuova
  versione del piano e nuovi hash.
- Le quote rare introducono intenzionalmente dipendenze tra camera, ruota e
  severita. Questo e preferibile alla mancanza di supporto nell'evaluation, ma
  le metriche finali dovranno essere riportate anche per sottogruppo.
- Gli override semantici paired sono ammessi soltanto con pilot disabilitato,
  devono seguire esattamente l'ordine degli indici e vengono validati
  fail-closed.
- Gli split qui presenti sono la fonte pre-render. Il futuro assemblatore del
  pool deve conservarli invece di ricalcolarli casualmente a posteriori.
