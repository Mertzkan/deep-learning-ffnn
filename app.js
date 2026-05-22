const STORAGE_DATA_KEY = 'ffnnRegressionDataset';
const MODEL_CLEAN_KEY = 'localstorage://ffnn-clean-model';
const MODEL_BEST_KEY = 'localstorage://ffnn-best-model';
const MODEL_OVERFIT_KEY = 'localstorage://ffnn-overfit-model';

const state = {
  data: null,
  models: {
    clean: null,
    best: null,
    overfit: null,
  },
  histories: {
    clean: null,
    best: null,
    overfit: null,
  },
  losses: {},
};

function setStatus(message) {
  document.getElementById('status').textContent = message;
}

function updateStorageButtons() {
  const hasDataset = localStorage.getItem(STORAGE_DATA_KEY) !== null;

  const hasCleanModel =
    localStorage.getItem('tensorflowjs_models/ffnn-clean-model/info') !== null;
  const hasBestModel =
    localStorage.getItem('tensorflowjs_models/ffnn-best-model/info') !== null;
  const hasOverfitModel =
    localStorage.getItem('tensorflowjs_models/ffnn-overfit-model/info') !==
    null;

  const hasAllModels = hasCleanModel && hasBestModel && hasOverfitModel;

  const loadDataButton = document.getElementById('loadDataBtn');
  const loadModelsButton = document.getElementById('loadModelsBtn');

  loadDataButton.disabled = !hasDataset;
  loadModelsButton.disabled = !hasAllModels;

  loadDataButton.title = hasDataset
    ? 'Gespeicherten Datensatz laden'
    : 'Noch kein gespeicherter Datensatz vorhanden';

  loadModelsButton.title = hasAllModels
    ? 'Gespeicherte Modelle laden'
    : 'Noch keine gespeicherten Modelle vorhanden';
}

function showPlotLoadingMessages() {
  const plotIds = [
    'plotDataClean',
    'plotDataNoisy',
    'plotLossClean',
    'plotLossNoisy',
    'plotCleanTrain',
    'plotCleanTest',
    'plotBestTrain',
    'plotBestTest',
    'plotOverfitTrain',
    'plotOverfitTest',
  ];

  for (const id of plotIds) {
    const element = document.getElementById(id);

    if (element) {
      element.innerHTML = `
        <div class="plotLoading">
          <strong>Diagramm wird vorbereitet</strong>
          <span>Daten werden erzeugt und visualisiert.</span>
        </div>
      `;
    }
  }
}

function showPredictionLoadingMessages() {
  const plotIds = [
    'plotLossClean',
    'plotLossNoisy',
    'plotCleanTrain',
    'plotCleanTest',
    'plotBestTrain',
    'plotBestTest',
    'plotOverfitTrain',
    'plotOverfitTest',
  ];

  for (const id of plotIds) {
    const element = document.getElementById(id);

    if (element) {
      element.innerHTML = `
        <div class="plotLoading">
          <strong>Training läuft</strong>
          <span>Modelle werden trainiert. Ergebnisse erscheinen automatisch.</span>
        </div>
      `;
    }
  }
}

function showLossUnavailableMessage() {
  const lossPlotIds = ['plotLossClean', 'plotLossNoisy'];

  for (const id of lossPlotIds) {
    const element = document.getElementById(id);

    if (element) {
      element.innerHTML = `
        <div class="plotLoading">
          <strong>Kein Loss Verlauf verfügbar</strong>
          <span>Die Modelle wurden geladen. TensorFlow.js speichert Modellgewichte, aber keine Trainingshistorie.</span>
        </div>
      `;
    }
  }
}

/*
  Seedbarer Zufallsgenerator.
  Dadurch sind Experimente reproduzierbar, was für Dokumentation und Bewertung wichtig ist.
*/
function mulberry32(seed) {
  let value = seed >>> 0;

  return function random() {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/*
  Box Muller Transformation für normalverteiltes Rauschen.
  Rückgabewert hat ungefähr Mittelwert 0 und Varianz 1.
*/
function gaussianRandom(random) {
  let u = 0;
  let v = 0;

  while (u === 0) u = random();
  while (v === 0) v = random();

  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/*
  Ground Truth Funktion.
  In der realen Simulation wird angenommen, dass das Modell diese Funktion nicht kennt.
  Sie dient nur zum Erzeugen der Trainings und Testdaten.
*/
function groundTruth(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function validateNumber(value, name) {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} ist keine gültige Zahl.`);
  }
}

function validateDataset(data) {
  const groups = ['trainClean', 'testClean', 'trainNoisy', 'testNoisy'];

  for (const group of groups) {
    if (!Array.isArray(data[group])) {
      throw new Error(`Datensatzfehler: ${group} fehlt.`);
    }

    for (const item of data[group]) {
      validateNumber(item.x, `${group}.x`);
      validateNumber(item.y, `${group}.y`);
    }
  }
}

function generateDataset() {
  const seed = Number(document.getElementById('seedInput').value);
  const n = Number(document.getElementById('nInput').value);
  const variance = Number(document.getElementById('noiseInput').value);

  validateNumber(seed, 'Seed');
  validateNumber(n, 'N');
  validateNumber(variance, 'Rauschvarianz');

  if (n < 20 || n % 2 !== 0) {
    throw new Error('N muss gerade und mindestens 20 sein.');
  }

  if (variance < 0) {
    throw new Error('Die Varianz darf nicht negativ sein.');
  }

  const random = mulberry32(seed);
  const standardDeviation = Math.sqrt(variance);
  const points = [];

  for (let i = 0; i < n; i++) {
    const x = -2 + 4 * random();
    const y = groundTruth(x);
    points.push({ x, y });
  }

  shuffleInPlace(points, random);

  const split = n / 2;
  const trainClean = points.slice(0, split).map((point) => ({ ...point }));
  const testClean = points.slice(split).map((point) => ({ ...point }));

  const trainNoisy = trainClean.map((point) => ({
    x: point.x,
    y: point.y + standardDeviation * gaussianRandom(random),
  }));

  const testNoisy = testClean.map((point) => ({
    x: point.x,
    y: point.y + standardDeviation * gaussianRandom(random),
  }));

  const grid = createPredictionGrid();

  const dataset = {
    seed,
    n,
    variance,
    trainClean,
    testClean,
    trainNoisy,
    testNoisy,
    grid,
  };

  validateDataset(dataset);
  return dataset;
}

function shuffleInPlace(array, random) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function createPredictionGrid() {
  const grid = [];

  for (let i = 0; i <= 250; i++) {
    const x = -2 + (4 * i) / 250;
    grid.push({ x, y: groundTruth(x) });
  }

  return grid;
}

function createModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [1],
      units: 100,
      activation: 'relu',
    }),
  );

  model.add(
    tf.layers.dense({
      units: 100,
      activation: 'relu',
    }),
  );

  model.add(
    tf.layers.dense({
      units: 1,
    }),
  );

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError',
  });

  return model;
}

function compileModel(model) {
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError',
  });

  return model;
}

function pointsToTensors(points) {
  const xs = tf.tensor2d(points.map((point) => [point.x]));
  const ys = tf.tensor2d(points.map((point) => [point.y]));

  return { xs, ys };
}

async function trainModel(model, trainPoints, testPoints, epochs, label) {
  const train = pointsToTensors(trainPoints);
  const test = pointsToTensors(testPoints);

  const losses = [];
  const testLosses = [];

  await model.fit(train.xs, train.ys, {
    epochs,
    batchSize: 32,
    shuffle: true,

    /*
      validationData wird hier nur beobachtet und nicht zur Modelloptimierung verwendet.
      Es ersetzt kein echtes Validierungsset.
    */
    validationData: [test.xs, test.ys],

    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        losses.push(logs.loss);
        testLosses.push(logs.val_loss);

        const isLogEpoch =
          epoch === 0 || (epoch + 1) % 50 === 0 || epoch + 1 === epochs;

        if (isLogEpoch) {
          setStatus(
            `${label}: Epoche ${epoch + 1} von ${epochs}, ` +
              `Train Loss ${logs.loss.toFixed(5)}, Test Loss ${logs.val_loss.toFixed(5)}`,
          );

          await tf.nextFrame();
        }
      },
    },
  });

  train.xs.dispose();
  train.ys.dispose();
  test.xs.dispose();
  test.ys.dispose();

  return {
    loss: losses,
    testLoss: testLosses,
  };
}

function predictPoints(model, points) {
  return tf.tidy(() => {
    const xs = tf.tensor2d(points.map((point) => [point.x]));
    const predictions = model.predict(xs);
    const values = Array.from(predictions.dataSync());

    return points.map((point, index) => ({
      x: point.x,
      y: values[index],
    }));
  });
}

function calculateMSE(model, points) {
  return tf.tidy(() => {
    const xs = tf.tensor2d(points.map((point) => [point.x]));
    const ys = tf.tensor2d(points.map((point) => [point.y]));
    const predictions = model.predict(xs);
    const mse = tf.metrics.meanSquaredError(ys, predictions).mean();

    return mse.dataSync()[0];
  });
}

function sorted(points) {
  return [...points].sort((a, b) => a.x - b.x);
}

function scatterTrace(points, name, color) {
  return {
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    type: 'scatter',
    mode: 'markers',
    name,
    marker: {
      color,
      size: 8,
      opacity: 0.85,
    },
  };
}

function lineTrace(points, name, color) {
  const ordered = sorted(points);

  return {
    x: ordered.map((point) => point.x),
    y: ordered.map((point) => point.y),
    type: 'scatter',
    mode: 'lines',
    name,
    line: {
      color,
      width: 3,
    },
  };
}

function baseLayout(title) {
  return {
    title,
    margin: {
      l: 52,
      r: 18,
      t: 45,
      b: 45,
    },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    xaxis: {
      title: 'x',
      range: [-2.1, 2.1],
      zeroline: true,
    },
    yaxis: {
      title: 'y',
      zeroline: true,
    },
    legend: {
      orientation: 'h',
      y: -0.22,
    },
  };
}

function renderDatasetPlots() {
  const data = state.data;

  Plotly.newPlot(
    'plotDataClean',
    [
      scatterTrace(data.trainClean, 'Train', '#1f77b4'),
      scatterTrace(data.testClean, 'Test', '#ff7f0e'),
      lineTrace(data.grid, 'Ground Truth', '#2ca02c'),
    ],
    baseLayout('Unverrauschter Datensatz'),
    { responsive: true },
  );

  Plotly.newPlot(
    'plotDataNoisy',
    [
      scatterTrace(data.trainNoisy, 'Train', '#1f77b4'),
      scatterTrace(data.testNoisy, 'Test', '#ff7f0e'),
      lineTrace(data.grid, 'Ground Truth nur zur Kontrolle', '#2ca02c'),
    ],
    baseLayout('Verrauschter Datensatz'),
    { responsive: true },
  );
}

function renderLossPlots() {
  const clean = state.histories.clean;
  const best = state.histories.best;
  const overfit = state.histories.overfit;

  Plotly.newPlot(
    'plotLossClean',
    [
      {
        x: clean.loss.map((_, index) => index + 1),
        y: clean.loss,
        type: 'scatter',
        mode: 'lines',
        name: 'Train Loss',
      },
      {
        x: clean.testLoss.map((_, index) => index + 1),
        y: clean.testLoss,
        type: 'scatter',
        mode: 'lines',
        name: 'Test Loss',
      },
    ],
    {
      ...baseLayout('Loss Clean Modell'),
      xaxis: { title: 'Epoche' },
      yaxis: { title: 'MSE', type: 'log' },
    },
    { responsive: true },
  );

  Plotly.newPlot(
    'plotLossNoisy',
    [
      {
        x: best.loss.map((_, index) => index + 1),
        y: best.loss,
        type: 'scatter',
        mode: 'lines',
        name: 'Best Train',
      },
      {
        x: best.testLoss.map((_, index) => index + 1),
        y: best.testLoss,
        type: 'scatter',
        mode: 'lines',
        name: 'Best Test',
      },
      {
        x: overfit.loss.map((_, index) => index + 1),
        y: overfit.loss,
        type: 'scatter',
        mode: 'lines',
        name: 'Overfit Train',
      },
      {
        x: overfit.testLoss.map((_, index) => index + 1),
        y: overfit.testLoss,
        type: 'scatter',
        mode: 'lines',
        name: 'Overfit Test',
      },
    ],
    {
      ...baseLayout('Loss Noisy Modelle'),
      xaxis: { title: 'Epoche' },
      yaxis: { title: 'MSE', type: 'log' },
    },
    { responsive: true },
  );
}

function renderPredictionPlot(elementId, title, model, points, showTruth) {
  const prediction = predictPoints(model, state.data.grid);

  const traces = [scatterTrace(points, 'Daten', '#1f77b4')];

  if (showTruth) {
    traces.push(lineTrace(state.data.grid, 'Ground Truth', '#2ca02c'));
  }

  traces.push(lineTrace(prediction, 'Modellvorhersage', '#d62728'));

  Plotly.newPlot(elementId, traces, baseLayout(title), { responsive: true });
}

function formatLoss(value) {
  return value.toFixed(6);
}

function renderAllPredictions() {
  const data = state.data;
  const models = state.models;

  state.losses.cleanTrain = calculateMSE(models.clean, data.trainClean);
  state.losses.cleanTest = calculateMSE(models.clean, data.testClean);
  state.losses.bestTrain = calculateMSE(models.best, data.trainNoisy);
  state.losses.bestTest = calculateMSE(models.best, data.testNoisy);
  state.losses.overfitTrain = calculateMSE(models.overfit, data.trainNoisy);
  state.losses.overfitTest = calculateMSE(models.overfit, data.testNoisy);

  renderPredictionPlot(
    'plotCleanTrain',
    'Clean Modell auf Train',
    models.clean,
    data.trainClean,
    true,
  );
  renderPredictionPlot(
    'plotCleanTest',
    'Clean Modell auf Test',
    models.clean,
    data.testClean,
    true,
  );
  renderPredictionPlot(
    'plotBestTrain',
    'Best Fit Modell auf Train',
    models.best,
    data.trainNoisy,
    true,
  );
  renderPredictionPlot(
    'plotBestTest',
    'Best Fit Modell auf Test',
    models.best,
    data.testNoisy,
    true,
  );
  renderPredictionPlot(
    'plotOverfitTrain',
    'Overfit Modell auf Train',
    models.overfit,
    data.trainNoisy,
    true,
  );
  renderPredictionPlot(
    'plotOverfitTest',
    'Overfit Modell auf Test',
    models.overfit,
    data.testNoisy,
    true,
  );

  document.getElementById('lossCleanTrain').textContent =
    `MSE Train: ${formatLoss(state.losses.cleanTrain)}`;
  document.getElementById('lossCleanTest').textContent =
    `MSE Test: ${formatLoss(state.losses.cleanTest)}`;
  document.getElementById('lossBestTrain').textContent =
    `MSE Train: ${formatLoss(state.losses.bestTrain)}`;
  document.getElementById('lossBestTest').textContent =
    `MSE Test: ${formatLoss(state.losses.bestTest)}`;
  document.getElementById('lossOverfitTrain').textContent =
    `MSE Train: ${formatLoss(state.losses.overfitTrain)}`;
  document.getElementById('lossOverfitTest').textContent =
    `MSE Test: ${formatLoss(state.losses.overfitTest)}`;
  renderExperimentSummary();
}

async function runExperiment(useExistingData = false) {
  const runButton = document.getElementById('runBtn');
  runButton.disabled = true;
  showPlotLoadingMessages();

  try {
    setStatus('Datensatz wird vorbereitet.');

    if (!useExistingData || !state.data) {
      state.data = generateDataset();
    }

    validateDataset(state.data);
    renderDatasetPlots();
    showPredictionLoadingMessages();

    const cleanEpochs = Number(
      document.getElementById('cleanEpochsInput').value,
    );
    const bestEpochs = Number(document.getElementById('bestEpochsInput').value);
    const overfitEpochs = Number(
      document.getElementById('overfitEpochsInput').value,
    );

    state.models.clean = createModel();
    state.models.best = createModel();
    state.models.overfit = createModel();

    setStatus('Trainiere Clean Modell.');
    state.histories.clean = await trainModel(
      state.models.clean,
      state.data.trainClean,
      state.data.testClean,
      cleanEpochs,
      'Clean Modell',
    );

    setStatus('Trainiere Best Fit Modell.');
    state.histories.best = await trainModel(
      state.models.best,
      state.data.trainNoisy,
      state.data.testNoisy,
      bestEpochs,
      'Best Fit Modell',
    );

    setStatus('Trainiere Overfit Modell.');
    state.histories.overfit = await trainModel(
      state.models.overfit,
      state.data.trainNoisy,
      state.data.testNoisy,
      overfitEpochs,
      'Overfit Modell',
    );

    renderLossPlots();
    renderAllPredictions();

    setStatus(
      'Fertig. Prüfe besonders die MSE Werte von Best Fit und Overfit auf Train und Test.',
    );
  } catch (error) {
    console.error(error);
    setStatus(`Fehler: ${error.message}`);
  } finally {
    runButton.disabled = false;
  }
}

function saveDataset() {
  if (!state.data) {
    setStatus('Es gibt noch keinen Datensatz zum Speichern.');
    return;
  }

  localStorage.setItem(STORAGE_DATA_KEY, JSON.stringify(state.data));
  updateStorageButtons();
  setStatus('Datensatz wurde im Local Storage gespeichert.');
}

function loadDataset() {
  const raw = localStorage.getItem(STORAGE_DATA_KEY);

  if (!raw) {
    setStatus('Kein gespeicherter Datensatz gefunden.');
    return;
  }

  const loaded = JSON.parse(raw);
  validateDataset(loaded);

  state.data = loaded;
  renderDatasetPlots();

  const hasModels =
    state.models.clean && state.models.best && state.models.overfit;

  if (hasModels) {
    renderAllPredictions();
    setStatus(
      'Datensatz wurde geladen und mit den aktuell geladenen Modellen ausgewertet.',
    );
  } else {
    setStatus(
      'Datensatz wurde geladen. Lade gespeicherte Modelle oder starte ein neues Training.',
    );
  }
}

async function saveModels() {
  if (!state.models.clean || !state.models.best || !state.models.overfit) {
    setStatus('Es gibt noch keine Modelle zum Speichern.');
    return;
  }

  await state.models.clean.save(MODEL_CLEAN_KEY);
  await state.models.best.save(MODEL_BEST_KEY);
  await state.models.overfit.save(MODEL_OVERFIT_KEY);

  updateStorageButtons();
  setStatus('Modelle wurden im Local Storage gespeichert.');
}

async function loadModels() {
  if (!state.data) {
    const raw = localStorage.getItem(STORAGE_DATA_KEY);

    if (raw) {
      const loaded = JSON.parse(raw);
      validateDataset(loaded);
      state.data = loaded;
    } else {
      state.data = generateDataset();
    }
  }

  validateDataset(state.data);
  renderDatasetPlots();

  state.models.clean = compileModel(await tf.loadLayersModel(MODEL_CLEAN_KEY));
  state.models.best = compileModel(await tf.loadLayersModel(MODEL_BEST_KEY));
  state.models.overfit = compileModel(
    await tf.loadLayersModel(MODEL_OVERFIT_KEY),
  );

  showLossUnavailableMessage();
  renderAllPredictions();

  setStatus(
    'Gespeicherte Modelle wurden geladen und ohne erneutes Training ausgewertet.',
  );
}

function renderExperimentSummary() {
  const cleanEpochs = document.getElementById('cleanEpochsInput').value;
  const bestEpochs = document.getElementById('bestEpochsInput').value;
  const overfitEpochs = document.getElementById('overfitEpochsInput').value;

  document.getElementById('experimentSettings').innerHTML =
    `Seed = ${state.data.seed}<br>` +
    `N = ${state.data.n}<br>` +
    `Rauschvarianz = ${state.data.variance}<br>` +
    `Clean Epochs = ${cleanEpochs}<br>` +
    `Best Fit Epochs = ${bestEpochs}<br>` +
    `Overfit Epochs = ${overfitEpochs}`;

  document.getElementById('experimentLosses').innerHTML =
    `Clean Train MSE = ${formatLoss(state.losses.cleanTrain)}<br>` +
    `Clean Test MSE = ${formatLoss(state.losses.cleanTest)}<br>` +
    `Best Fit Train MSE = ${formatLoss(state.losses.bestTrain)}<br>` +
    `Best Fit Test MSE = ${formatLoss(state.losses.bestTest)}<br>` +
    `Overfit Train MSE = ${formatLoss(state.losses.overfitTrain)}<br>` +
    `Overfit Test MSE = ${formatLoss(state.losses.overfitTest)}`;

  const bestTestIsBetter = state.losses.bestTest < state.losses.overfitTest;
  const overfitTrainIsBetter =
    state.losses.overfitTrain < state.losses.bestTrain;

  if (bestTestIsBetter && overfitTrainIsBetter) {
    document.getElementById('experimentObservation').textContent =
      'Beobachtung: Das Overfit Modell erreicht den kleineren Trainings MSE, generalisiert aber schlechter als das Best Fit Modell. Das erkennt man am höheren Test MSE.';
  } else {
    document.getElementById('experimentObservation').textContent =
      'Beobachtung: Die Ergebnisse zeigen den Zusammenhang zwischen Trainingsdauer, Trainings MSE und Test MSE. Für eine klare Overfitting Darstellung können die Epochen weiter angepasst werden.';
  }
}

function registerEventListeners() {
  document
    .getElementById('runBtn')
    .addEventListener('click', () => runExperiment(false));
  document.getElementById('saveDataBtn').addEventListener('click', saveDataset);
  document.getElementById('loadDataBtn').addEventListener('click', loadDataset);
  document
    .getElementById('saveModelsBtn')
    .addEventListener('click', saveModels);

  document.getElementById('loadModelsBtn').addEventListener('click', () => {
    loadModels().catch((error) => {
      setStatus(`Fehler beim Laden der Modelle: ${error.message}`);
    });
  });
}

window.addEventListener('load', () => {
  registerEventListeners();
  updateStorageButtons();
  showPlotLoadingMessages();
  runExperiment(false);
});
