# Codex repository guide

Il repository contiene due progetti indipendenti:

- `semantic_segmentation/` per la segmentazione semantica S5Mars;
- `wheel_anomaly_detection/` per Blender, generazione dataset e visual anomaly
  detection delle ruote.

Eseguire comandi, test e notebook dalla directory del progetto interessato.

Prima di lavorare su file, config, script o output della pipeline Blender,
leggere `wheel_anomaly_detection/docs/generation/INDEX.md` e usare il suo
instradamento per selezionare la documentazione di milestone pertinente.

I file sotto `wheel_anomaly_detection/docs/archive/legacy_blender/` sono un
archivio storico. Non aggiornarli per milestone ordinarie: aggiornare invece
`docs/generation/INDEX.md` e il documento di milestone pertinente.

## Confini del repository

- Non mescolare il codice di `semantic_segmentation/` con la pipeline ruote.
- Gli entrypoint host Blender appartengono a
  `wheel_anomaly_detection/scripts/host/`; non devono importare `bpy`.
- Gli entrypoint Blender appartengono a
  `wheel_anomaly_detection/scripts/blender/`.
- La logica pura della generazione vive nei package esistenti sotto
  `wheel_anomaly_detection/src/`.
- Il codice ML autorevole vive in
  `wheel_anomaly_detection/src/anomaly_detection/`; il notebook Kaggle è
  autosufficiente e duplica esplicitamente il codice necessario, così da non
  dipendere dai sorgenti del repository. Moduli e notebook devono restare
  sincronizzati.
- Le configurazioni Blender restano in
  `wheel_anomaly_detection/configs/blender/`; quelle ML stanno in
  `wheel_anomaly_detection/configs/anomaly_detection/`.
- Gli asset originali sono immutabili, locali e ignorati sotto
  `wheel_anomaly_detection/assets/original/`. Non modificare, convertire,
  scaricare o committare il GLB NASA.
- Output e log appartengono alle directory locali del rispettivo progetto e
  non devono essere committati.

## Contratto Blender corrente

- Target riproducibile: Blender 5.2.0 LTS, Eevee, 800x600, 4:3.
- Asset reale: `24584_Curiosity_static.glb`, SHA-256
  `86a8ee6d39fb1711ae549ba99951d2f52b44a7b8367ce4814cc81aec60e94b31`.
- Verdict audit B: sei ruote separabili automaticamente dal mesh `MSL`.
- Default fase 2: `wheel_candidate_05`; non cambiare candidato silenziosamente.
- Operare solo su copie importate e verificare il checksum prima e dopo.
- Nessuna dipendenza ML dentro Blender Python.

## Verifica

Dalla directory `wheel_anomaly_detection`:

```powershell
python -m unittest discover -s tests -p 'test_blender_audit*.py' -v
python -m unittest tests.test_wheel_preparation_core tests.test_wheel_preparation_validation -v
```

Per modifiche alla generazione, eseguire inoltre il real GLB pipeline e il
validator indicati nel documento di milestone. Il JSON non sostituisce la QA
visiva dei render richiesti.

Preservare modifiche utente non correlate e sincronizzare la documentazione
quando cambia un gate, fallback, schema output o requisito Blender.

## Critical reanalysis rule

Prima di implementare un design, eseguire una seconda revisione avversariale:
valutare failure mode, effetti collaterali, rischi di scala/prestazioni e rischi
di validità dei dati; correggere l'approccio e documentare i trade-off
materiali.
