## Guida operativa prevista

### Movimento manuale

1. Apri la scena `Simulator`.
2. Attiva il pannello “Rover Controller”.
3. Premi “Start simulation”.
4. Usa:
   - `W/S` per avanzare e retrocedere;
   - `A/D` per sterzare;
   - `Shift` per aumentare la velocità;
   - mouse per regolare la camera;
   - `C` per alternare camera di guida e camera della ruota.
5. Il controller esegue continuamente i quattro raycast.
6. Se manca un contatto o la pendenza supera il limite, il movimento viene annullato.
7. Passa alla camera della ruota e controlla inquadratura, terreno e occlusioni.
8. Premi `P` per salvare la posa.
9. La posa viene aggiunta a un JSON con posizione, rotazione, contatti, camera e identificatore del terreno.
10. Da quella posa genera la coppia sana/anomala usando camera e illuminazione identiche.

### Posizionamenti automatici

1. Guida manualmente il rover in una zona interessante.
2. Premi “Generate nearby poses”.
3. Imposta:
   - numero di pose;
   - raggio intorno alla posizione corrente;
   - intervallo di rotazione;
   - pendenza massima;
   - seed.
4. Il sistema campiona posizione e orientamento.
5. Per ogni candidato esegue i quattro raycast.
6. Scarta automaticamente pose:
   - fuori dal terreno;
   - troppo inclinate;
   - con contatti mancanti;
   - con la ruota fuori dall’immagine;
   - con anomalie troppo piccole o occluse.
7. Mostra le pose valide per un controllo manuale.
8. Le pose approvate vengono aggiunte allo stesso JSON.
9. Il generatore produce una coppia sana/anomala per ogni posa.