# Bulk sequence 005 — 2.500 immagini

## Stato operativo

Avviata il 17 agosto 2026 come sequenza seriale resumable. La composizione usa
esclusivamente unita pending del piano canonico e produce esattamente 2.500
immagini:

- `wave_005_clean_remaining_1150`: 650 clean train e 500 clean validation;
- `wave_006_pairs_675`: 250 coppie validation e 425 coppie test, equivalenti
  a 1.350 RGB;
- totale: 1.150 clean + 1.350 immagini paired = 2.500 immagini.

La sequenza completa train e validation. Dopo il suo completamento resteranno
soltanto 550 coppie test, cioe 1.100 immagini, per raggiungere il dataset da
10.000 immagini.

## Esecuzione e monitoraggio

- config sequenza: `configs/blender/bulk_sequences/sequence_005_2500.json`;
- stato: `outputs/anomaly_detection_2/bulk/sequences/sequence_005_2500/status.json`;
- log: `logs/anomaly_detection_2/bulk/sequence_005_2500.*.log`;
- PID: `logs/anomaly_detection_2/bulk/sequence_005_2500.pid`;
- i due renderer sono seriali per evitare contesa sulla stessa GPU;
- ogni step riparte automaticamente in resume se la directory del run esiste.

La stima iniziale, basata sulle wave clean e sul pre-bulk paired misurati, e
circa 7 h 15 min complessive: circa 2 h 14 min clean e circa 5 h paired.

Il completamento deve essere seguito dai due validator e dall'aggiornamento
del pool canonico; lo stato `complete` della sequenza da solo non sostituisce
questi controlli.

## Ripartenza paired

Il primo avvio paired si e fermato su `dr_200001`: tutti i gate geometrici e
di localita erano validi, ma la mediana della differenza RGB era `9/255`
contro la soglia `10/255`. Il retry identico ha confermato che non era un crash
casuale. Poiche l'86,44% della maschera superava gia la soglia di cambiamento
`6/255`, la soglia mediana produttiva e stata portata a `9/255`: mantiene un
vincolo fotometrico piu severo del gate per-pixel, ma evita che un singolo
livello di quantizzazione blocchi il piano deterministico.

Il secondo tentativo si e fermato su `dr_200011`, con mediana `8/255`, pur
avendo l'86,57% della maschera sopra `6/255` e tutti gli altri gate validi.
Questo ha dimostrato che correggere il limite di un singolo livello non era
robusto. Il limite mediano e quindi fissato a `6/255`, uguale alla soglia del
pixel cambiato: con la quota minima del 65% sopra tale soglia, il gate mediano
resta verificato matematicamente e non puo contraddire il gate principale.
Area, ROI, componente unica, localita dell'energia ed effetto massimo restano
fail-closed e invariati.

Su richiesta, entrambe le directory dei tentativi falliti sono state rimosse.
Il run definitivo riparte da zero come `wave_006_pairs_675_final`.

Anche questo tentativo si e fermato su `dr_200012`: la frazione cambiata era
`0,6394`, appena sotto il precedente minimo `0,65`, mentre mediana e ogni altro
gate erano validi. Su decisione esplicita il minimo produttivo e ora `0,58`.
La run fallita da 18 coppie e stata rimossa e la generazione riparte da zero
come `wave_006_pairs_675_cf58`.

Il run `cf58` ha poi raggiunto 49 coppie prima di fermarsi su `dr_200060`, un
foro strutturalmente valido ma a basso contrasto (`changed_fraction=0,3937`,
mediana `4/255`). Questo ha confermato che i limiti fotometrici per-campione
non sono adatti come condizione di arresto del bulk: luce, polvere e sfondo
possono produrre casi difficili ma correttamente etichettati.

Il contratto produttivo separa ora:

- gate bloccanti di geometria, maschera, ROI, componente, localita, area ed
  effetto non nullo;
- metriche diagnostiche `changed_fraction` e `median_delta`;
- stato `normal_contrast` oppure `low_contrast` registrato nel manifest;
- stato `no_effect`, che resta bloccante.

Il validator accetta i casi `low_contrast`, ne riporta il conteggio e continua
a rifiutare maschere o geometrie invalide e coppie prive di differenza. La run
`cf58` viene rimossa e il nuovo run da zero e
`wave_006_pairs_675_photometric_diagnostic`.

Dopo un arresto del PC durante il primo chunk, il run diagnostico e stato
riavviato con `--resume`. Al momento del crash il manifest non conteneva
coppie committate; tre artefatti orfani per pass erano presenti nei path
finali e vengono rigenerati deterministicamente. Fingerprint, piano, config e
codice coincidevano, quindi non e stato necessario cancellare il run.
