import React, { useState, useEffect, useRef, useCallback } from 'react';
import { MLP, Value } from 'nngrad';

// Custom sigmoid function using Value.
const sigmoid = (x) => {
  const s = 1 / (1 + Math.exp(-x.data));
  const out = new Value(s, [x], 'sigmoid');
  out._backward = function () {
    x.grad += out.data * (1 - out.data) * out.grad;
  };
  return out;
};

// Dataset generators (spiral, circle, moon)
const generateSpiralData = (samples = 100, classes = 2) => {
  const points = [];
  const labels = [];
  for (let i = 0; i < classes; i++) {
    const r = 5;
    const theta0 = (i * 2 * Math.PI) / classes;
    for (let j = 0; j < samples; j++) {
      const theta = theta0 + (j / samples) * 4 * Math.PI;
      const radius = (r * j) / samples;
      const x = radius * Math.sin(theta);
      const y = radius * Math.cos(theta);
      const nx = x + Math.random() * 0.5 - 0.25;
      const ny = y + Math.random() * 0.5 - 0.25;
      points.push([nx, ny]);
      labels.push(i);
    }
  }
  return { points, labels };
};

const generateCircleData = (samples = 100) => {
  const points = [];
  const labels = [];
  for (let i = 0; i < samples; i++) {
    const theta = Math.random() * 2 * Math.PI;
    const radius = Math.random() * 2;
    points.push([radius * Math.cos(theta), radius * Math.sin(theta)]);
    labels.push(0);
  }
  for (let i = 0; i < samples; i++) {
    const theta = Math.random() * 2 * Math.PI;
    const radius = 3 + Math.random();
    points.push([radius * Math.cos(theta), radius * Math.sin(theta)]);
    labels.push(1);
  }
  return { points, labels };
};

const generateMoonData = (samples = 100) => {
  const points = [];
  const labels = [];
  for (let i = 0; i < samples; i++) {
    const theta = Math.PI + Math.random() * Math.PI;
    const radius = 4;
    const x = radius * Math.cos(theta);
    const y = radius * Math.sin(theta) + Math.random() - 0.5;
    points.push([x, y]);
    labels.push(0);
  }
  for (let i = 0; i < samples; i++) {
    const theta = Math.random() * Math.PI;
    const radius = 4;
    const x = radius * Math.cos(theta);
    const y = radius * Math.sin(theta) + Math.random() - 0.5;
    points.push([x, y]);
    labels.push(1);
  }
  return { points, labels };
};

// Color interpolation (maps a value in [-1,1] to a color)
const interpolateColor = (value) => {
  const t = (value + 1) / 2;
  const r = Math.round(255 * (1 - t));
  const b = Math.round(255 * t);
  const g = Math.round(100 * Math.min(1 - Math.abs(t - 0.5) * 2, 0.5));
  return `rgba(${r}, ${g}, ${b}, 0.5)`;
};

const NeuralNetworkVisualizer = () => {
  // UI state for dataset, training stats, hyperparameters, etc.
  const [dataset, setDataset] = useState('spiral');
  const [isTraining, setIsTraining] = useState(false);
  const [epochs, setEpochs] = useState(0);
  const [loss, setLoss] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [learningRate, setLearningRate] = useState(0.001);
  const [hiddenLayerSizes, setHiddenLayerSizes] = useState([10, 10]);
  const [dataPoints, setDataPoints] = useState({ points: [], labels: [] });

  // Refs for canvas, animation frame, model, training flag, and training accumulators.
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const modelRef = useRef(null);
  const isTrainingRef = useRef(false);
  const batchIndexRef = useRef(0);
  const epochLossRef = useRef(0);
  const epochCorrectRef = useRef(0);
  const epochCountRef = useRef(0);
  const BATCH_SIZE = 20; // Adjust mini-batch size to reduce per-frame work.

  // Sync training flag ref with state.
  useEffect(() => {
    isTrainingRef.current = isTraining;
  }, [isTraining]);

  // Generate dataset when the dataset selection changes.
  useEffect(() => {
    generateData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  const generateData = () => {
    let data;
    switch (dataset) {
      case 'spiral':
        data = generateSpiralData(100, 2);
        break;
      case 'circle':
        data = generateCircleData(100);
        break;
      case 'moon':
        data = generateMoonData(100);
        break;
      default:
        data = generateSpiralData(100, 2);
    }
    setDataPoints(data);
    setEpochs(0);
    setLoss(0);
    setAccuracy(0);
    // Reset mini-batch accumulators.
    batchIndexRef.current = 0;
    epochLossRef.current = 0;
    epochCorrectRef.current = 0;
    epochCountRef.current = 0;
    initializeModel();
    const grid = generateModelData();
    drawVisualization(data.points, data.labels, grid);
  };

  const initializeModel = () => {
    // For challenging datasets like Spiral, consider a deeper network.
    let layers = [...hiddenLayerSizes, 1];
    modelRef.current = new MLP(2, layers);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const startTraining = () => {
    if (isTrainingRef.current) return;
    setIsTraining(true);
    isTrainingRef.current = true;
    // Reset epoch accumulators.
    batchIndexRef.current = 0;
    epochLossRef.current = 0;
    epochCorrectRef.current = 0;
    epochCountRef.current = 0;
    setEpochs(0);
    trainModel();
  };

  const stopTraining = () => {
    setIsTraining(false);
    isTrainingRef.current = false;
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  // Compute decision boundary grid (lower resolution to speed up drawing).
  const generateModelData = () => {
    if (!modelRef.current) return [];
    const resolution = 30;
    const xRange = [-6, 6];
    const yRange = [-6, 6];
    const xStep = (xRange[1] - xRange[0]) / resolution;
    const yStep = (yRange[1] - yRange[0]) / resolution;
    const grid = [];
    for (let i = 0; i <= resolution; i++) {
      for (let j = 0; j <= resolution; j++) {
        const x = xRange[0] + i * xStep;
        const y = yRange[0] + j * yStep;
        const rawOutput = modelRef.current.call([x, y])[0];
        const p = sigmoid(rawOutput).data;
        grid.push({ x, y, value: p });
      }
    }
    return grid;
  };

  // Mini-batch training loop.
  const trainModel = () => {
    if (!isTrainingRef.current) return;
    const model = modelRef.current;
    const { points, labels } = dataPoints;
    const totalSamples = points.length;

    // Process a mini-batch.
    const start = batchIndexRef.current;
    const end = Math.min(start + BATCH_SIZE, totalSamples);
    let miniBatchLoss = new Value(0);
    let miniBatchCorrect = 0;
    for (let i = start; i < end; i++) {
      const [x, y] = points[i];
      const target = labels[i]; // 0 or 1
      const rawOutput = model.call([x, y])[0];
      const p = sigmoid(rawOutput);
      const pred = p.data > 0.5 ? 1 : 0;
      if (pred === target) miniBatchCorrect++;
      // MSE loss for this sample.
      const diff = p.add(new Value(-target));
      const sampleLoss = diff.mul(diff);
      miniBatchLoss = miniBatchLoss.add(sampleLoss);
    }
    // Backpropagate mini-batch loss.
    miniBatchLoss.backward();
    // Update parameters (apply average gradient over the mini-batch).
    const params = model.parameters();
    for (const p of params) {
      p.data -= learningRate * p.grad;
    }
    // Reset gradients for next mini-batch.
    model.zero_grad();

    // Accumulate epoch statistics.
    const batchCount = end - start;
    epochLossRef.current += miniBatchLoss.data;
    epochCorrectRef.current += miniBatchCorrect;
    epochCountRef.current += batchCount;
    batchIndexRef.current = end;

    // If we've processed the entire dataset, end the epoch.
    if (batchIndexRef.current >= totalSamples) {
      const avgLoss = epochLossRef.current / epochCountRef.current;
      const acc = (epochCorrectRef.current / epochCountRef.current) * 100;
      setLoss(avgLoss);
      setAccuracy(acc);
      setEpochs((prev) => prev + 1);

      // Update visualization once per epoch.
      const grid = generateModelData();
      drawVisualization(points, labels, grid);

      // Stop if accuracy is at least 98%.
      if (acc >= 98) {
        stopTraining();
        console.log('Target accuracy reached:', acc);
        return;
      }
      // Reset for next epoch.
      batchIndexRef.current = 0;
      epochLossRef.current = 0;
      epochCorrectRef.current = 0;
      epochCountRef.current = 0;
    }

    // Schedule the next mini-batch.
    animationRef.current = requestAnimationFrame(trainModel);
  };

  // Visualization: Draw decision boundary grid and data points.
  const drawVisualization = useCallback((points, labels, grid) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    const margin = 20;
    const scale = Math.min((width - 2 * margin) / 12, (height - 2 * margin) / 12);
    const translateX = width / 2;
    const translateY = height / 2;

    // Draw decision boundary grid.
    if (grid && grid.length > 0) {
      for (const point of grid) {
        const x = translateX + point.x * scale;
        const y = translateY - point.y * scale;
        // Map sigmoid output [0,1] to a color scale (converted to [-1,1]).
        ctx.fillStyle = interpolateColor(point.value * 2 - 1);
        ctx.fillRect(x - scale / 2, y - scale / 2, scale, scale);
      }
    }
    // Draw data points.
    for (let i = 0; i < points.length; i++) {
      const [x, y] = points[i];
      const label = labels[i];
      const canvasX = translateX + x * scale;
      const canvasY = translateY - y * scale;
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 4, 0, 2 * Math.PI);
      ctx.fillStyle = label === 0 ? 'rgba(255, 100, 100, 0.8)' : 'rgba(100, 100, 255, 0.8)';
      ctx.fill();
      ctx.stroke();
    }
    // Draw coordinate axes.
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.moveTo(0, translateY);
    ctx.lineTo(width, translateY);
    ctx.moveTo(translateX, 0);
    ctx.lineTo(translateX, height);
    ctx.stroke();
  }, []);

  // Update canvas size on window resize.
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current) {
        canvasRef.current.width = canvasRef.current.parentElement.clientWidth;
        canvasRef.current.height = 400;
        const grid = generateModelData();
        drawVisualization(dataPoints.points, dataPoints.labels, grid);
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [dataPoints, drawVisualization]);

  // Update hidden layer configuration and reinitialize model.
  const updateHiddenLayers = (config) => {
    setHiddenLayerSizes(config);
    initializeModel();
    setEpochs(0);
    setLoss(0);
    setAccuracy(0);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-4 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Neural Network Training Visualizer</h1>
        <p className="text-gray-600">Watch a neural network learn to classify data in real-time</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="md:col-span-3 bg-white rounded-lg shadow p-4">
          <canvas
            ref={canvasRef}
            className="w-full h-96 border border-gray-200 rounded"
            width={800}
            height={400}
          />
          <div className="mt-4 flex flex-wrap justify-between items-center">
            <div className="space-x-2">
              <button
                onClick={startTraining}
                disabled={isTrainingRef.current}
                className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-blue-300"
              >
                Start Training
              </button>
              <button
                onClick={stopTraining}
                disabled={!isTrainingRef.current}
                className="px-4 py-2 bg-red-600 text-white rounded disabled:bg-red-300"
              >
                Stop Training
              </button>
              <button
                onClick={generateData}
                disabled={isTrainingRef.current}
                className="px-4 py-2 bg-gray-600 text-white rounded disabled:bg-gray-300"
              >
                Regenerate Data
              </button>
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <div>
                <span className="font-semibold">Dataset: </span>
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  disabled={isTrainingRef.current}
                  className="border rounded px-2 py-1"
                >
                  <option value="spiral">Spiral</option>
                  <option value="circle">Circle</option>
                  <option value="moon">Moon</option>
                </select>
              </div>
              <div>
                <span className="font-semibold">Learning Rate: </span>
                <select
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  disabled={isTrainingRef.current}
                  className="border rounded px-2 py-1"
                >
                  <option value="0.001">0.001</option>
                  <option value="0.005">0.005</option>
                  <option value="0.01">0.01</option>
                </select>
              </div>
              <div>
                <span className="font-semibold">Architecture: </span>
                <select
                  value={hiddenLayerSizes.join(',')}
                  onChange={(e) => updateHiddenLayers(e.target.value.split(',').map(Number))}
                  disabled={isTrainingRef.current}
                  className="border rounded px-2 py-1"
                >
                  <option value="5">5 neurons</option>
                  <option value="10">10 neurons</option>
                  <option value="5,5">5,5 neurons</option>
                  <option value="10,10">10,10 neurons</option>
                  <option value="20,10,5">20,10,5 neurons</option>
                </select>
              </div>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow p-4 space-y-4">
          <div>
            <h2 className="text-xl font-bold mb-2">Network Architecture</h2>
            <div className="space-y-2">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium">Input Layer</span>
                  <span className="text-sm font-medium">2 neurons</span>
                </div>
                <div className="w-full bg-blue-100 h-6 rounded flex items-center justify-center">
                  <span className="text-xs text-blue-800">Input [x, y]</span>
                </div>
              </div>
              {hiddenLayerSizes.map((size, index) => (
                <div key={index}>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Hidden Layer {index + 1}</span>
                    <span className="text-sm font-medium">{size} neurons</span>
                  </div>
                  <div className="w-full bg-purple-100 h-6 rounded flex items-center justify-center">
                    <span className="text-xs text-purple-800">ReLU Activation</span>
                  </div>
                </div>
              ))}
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium">Output Layer</span>
                  <span className="text-sm font-medium">1 neuron (Sigmoid)</span>
                </div>
                <div className="w-full bg-green-100 h-6 rounded flex items-center justify-center">
                  <span className="text-xs text-green-800">Sigmoid Activation</span>
                </div>
              </div>
            </div>
          </div>
          <div>
            <h2 className="text-xl font-bold mb-2">Training Stats</h2>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Epochs:</span>
                <span className="text-sm">{epochs}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Loss:</span>
                <span className="text-sm">{loss.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Accuracy:</span>
                <span className="text-sm">{accuracy.toFixed(2)}%</span>
              </div>
            </div>
          </div>
          <div>
            <h2 className="text-xl font-bold mb-2">Legend</h2>
            <div className="space-y-2">
              <div className="flex items-center">
                <div className="w-4 h-4 rounded-full bg-red-500 mr-2"></div>
                <span className="text-sm">Class 0</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded-full bg-blue-500 mr-2"></div>
                <span className="text-sm">Class 1</span>
              </div>
              <div className="flex items-center">
                <div className="w-16 h-4 bg-gradient-to-r from-red-500 to-blue-500 mr-2"></div>
                <span className="text-sm">Decision Boundary</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NeuralNetworkVisualizer;
