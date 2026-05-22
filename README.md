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

- N = 100
- Train = 50 Punkte
- Test = 50 Punkte
- Rauschvarianz = 0.05
- Optimizer = Adam
- Lernrate = 0.01
- Batch Size = 32
- Architektur = 2 Hidden Layer mit je 100 ReLU Neuronen
- Output Layer = 1 lineares Neuron
- Loss = Mean Squared Error
