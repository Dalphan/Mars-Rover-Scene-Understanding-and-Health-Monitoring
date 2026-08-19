# Handoff archive

Questa cartella e un archivio storico della pipeline Blender precedente e non
e piu il punto di ingresso per il lavoro corrente su `anomaly_detection_2`.

Per la pipeline corrente consultare prima
[`docs/anomaly_detection_2/INDEX.md`](../anomaly_detection_2/INDEX.md), quindi
il documento della milestone indicato dall'indice.

## Quando consultare questo archivio

- manutenzione della vecchia pipeline di preparazione/perforazione;
- ricostruzione di una decisione o di un fallimento storico;
- verifica di comandi, artefatti o checksum appartenenti ai run legacy;
- migrazione esplicitamente richiesta da una vecchia milestone.

I file principali dell'archivio sono:

- `blender_anomaly_detection.md`: diario tecnico e sequenza cronologica dei
  vecchi checkpoint Blender;
- `codex_context.json`: snapshot strutturato dello stato storico.

## Regola di manutenzione

Non duplicare qui le nuove milestone ordinarie di `anomaly_detection_2`.
Aggiornare invece il suo `INDEX.md` e il documento tecnico specifico. Questo
archivio va modificato soltanto quando cambia realmente una parte della
pipeline legacy o quando si esegue una migrazione storica esplicita.
