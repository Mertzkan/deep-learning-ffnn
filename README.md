# FFNN Regression mit TensorFlow.js

## Start in VS Code

1. Ordner in VS Code öffnen.
2. Extension `Live Server` installieren.
3. Rechtsklick auf `index.html`.
4. `Open with Live Server` auswählen.

Die Anwendung startet automatisch einen Trainingsdurchlauf.

## Dateien

- `index.html`: Struktur, Dokumentation und Diagrammbereiche.
- `style.css`: Layout, Farben und responsive Darstellung.
- `app.js`: Datenlogik, TensorFlow.js Modelle, Training, Evaluation, Plotly Visualisierung, Speichern und Laden.

## Standardparameter

- Seed = 42
- N = 100
- Train = 50 Punkte
- Test = 50 Punkte
- Intervall = [-2, 2]
- Rauschvarianz = 0.05
- Rauschen = normalverteilt, nur auf y Labels
- Optimizer = Adam
- Lernrate = 0.01
- Batch Size = 32
- Architektur = 2 Hidden Layer mit je 100 ReLU Neuronen
- Output Layer = 1 lineares Neuron
- Aktivierung Hidden Layer = ReLU
- Aktivierung Output Layer = linear
- Loss = Mean Squared Error
- Clean Epochs = 1200
- Best Fit Epochs = 300
- Overfit Epochs = 8000

## Finales Experiment

- Clean Train MSE = 0.000034
- Clean Test MSE = 0.009105
- Best Fit Train MSE = 0.035399
- Best Fit Test MSE = 0.066275
- Overfit Train MSE = 0.021361
- Overfit Test MSE = 0.069188
