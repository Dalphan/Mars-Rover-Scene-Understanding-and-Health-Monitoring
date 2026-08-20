# Notebook anomaly detection

Questa directory contiene il notebook end-to-end per Kaggle.

`kaggle_wheel_anomaly_detection.ipynb` è autosufficiente: contiene una copia
esplicita del codice per Dataset, preprocessing, audit e DataLoader e non importa
`src/anomaly_detection/`. Su Kaggle è quindi sufficiente collegare il dataset
già estratto.

La duplicazione è intenzionale, ma introduce il rischio che notebook e moduli
divergano. Ogni modifica al caricamento dei dati deve essere riportata in
entrambi. La configurazione corrente usa il resize diretto `256x256`, converte
sempre gli RGB in `float32` e usa la normalizzazione ImageNet. Le augmentation
train sono controllate dal modello e sono disabilitate per PatchCore. Una cella
di audit mostra il preprocessing effettivo. Tutti i parametri modificabili su
Kaggle, inclusi modello, metriche e upload Google Drive opzionale, sono raccolti
nella cella iniziale `Configuration`.
