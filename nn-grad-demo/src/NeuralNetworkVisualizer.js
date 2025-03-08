import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ReactFlow, { Controls, Background } from 'react-flow-renderer';
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

// -------------------------------
// Custom Edge Component to Animate Data Flow
// -------------------------------
const DataFlowEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  style,
  markerEnd,
}) => {
  const edgePath = `M${sourceX},${sourceY} L${targetX},${targetY}`;
  return (
    <>
      <path id={id} className="react-flow__edge-path" d={edgePath} style={style} markerEnd={markerEnd} />
      <circle cx={sourceX} cy={sourceY} r={4} fill="blue">
        <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
      </circle>
    </>
  );
};

// -------------------------------
// Advanced Network Architecture Visualizer
// -------------------------------
const NetworkArchitectureVisualizer = ({ layersSummary }) => {
  // If layersSummary is provided, build nodes from it.
  const nodes = useMemo(() => {
    if (!layersSummary || layersSummary.length === 0) return [];
    return layersSummary.map((layer, index) => ({
      id: `layer-${index}`,
      data: { 
        label: layer.layerName === 'Input Layer'
          ? `${layer.layerName}\n(${layer.neurons} neurons)\n${layer.label}`
          : `${layer.layerName}\nAvg Weight: ${layer.avgWeight.toFixed(2)}\nAvg Bias: ${layer.avgBias.toFixed(2)}\nActivation: ${layer.activation}`
      },
      position: { x: 50 + index * 220, y: 50 },
      style: {
        border: '2px solid ' + (layer.layerName === 'Input Layer' ? '#007AFF' : (layer.layerName === 'Output Layer' ? '#34C759' : '#FF9500')),
        padding: 10,
        borderRadius: 5,
        backgroundColor: '#fff',
        whiteSpace: 'pre-line'
      }
    }));
  }, [layersSummary]);

  const edges = useMemo(() => {
    const edgeArray = [];
    if (nodes.length < 2) return edgeArray;
    for (let i = 0; i < nodes.length - 1; i++) {
      edgeArray.push({
        id: `e-${i}-${i + 1}`,
        source: nodes[i].id,
        target: nodes[i + 1].id,
        type: 'dataflow', // use our custom edge
        animated: true,
      });
    }
    return edgeArray;
  }, [nodes]);

  const edgeTypes = useMemo(() => ({ dataflow: DataFlowEdge }), []);

  return (
    <div style={{ height: 300, border: '1px solid #ccc', borderRadius: 5 }}>
      <ReactFlow nodes={nodes} edges={edges} edgeTypes={edgeTypes} fitView>
        <Background gap={16} color="#aaa" />
        <Controls />
      </ReactFlow>
    </div>
  );
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
  // Advanced training options:
  const [batchSize, setBatchSize] = useState(20);
  const batchSizeRef = useRef(batchSize);
  const [enableLrDecay, setEnableLrDecay] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [chartTooltip, setChartTooltip] = useState(null);
  const [modelInsights, setModelInsights] = useState(null);
  const [modelSummary, setModelSummary] = useState([]);


  // Refs for canvases, animation frame, model, training flag, and accumulators.
  const canvasRef = useRef(null);
  const chartCanvasRef = useRef(null);
  const animationRef = useRef(null);
  const modelRef = useRef(null);
  const isTrainingRef = useRef(false);
  const batchIndexRef = useRef(0);
  const epochLossRef = useRef(0);
  const epochCorrectRef = useRef(0);
  const epochCountRef = useRef(0);

  // Synchronize training flag and batch size ref with state.
  useEffect(() => {
    isTrainingRef.current = isTraining;
  }, [isTraining]);

  useEffect(() => {
    batchSizeRef.current = batchSize;
  }, [batchSize]);

  // ------------------------------
  // Draw the training chart (with white background, grid, and curves).
  const drawTrainingChart = useCallback((history) => {
    if (!chartCanvasRef.current) return;
    const canvas = chartCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    // Fill the background.
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);

    const margin = 30;
    const chartWidth = width - 2 * margin;
    const chartHeight = height - 2 * margin;

    // Use the latest epoch number from history as maxEpoch.
    const maxEpoch = history.length > 0 ? history[history.length - 1].epoch : 1;
    const losses = history.map((d) => d.loss);
    const maxLoss = losses.length > 0 ? Math.max(...losses) : 1;

    // Draw grid lines.
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 5; i++) {
      const x = margin + (i / 5) * chartWidth;
      ctx.moveTo(x, margin);
      ctx.lineTo(x, height - margin);
    }
    for (let i = 0; i <= 5; i++) {
      const y = margin + (i / 5) * chartHeight;
      ctx.moveTo(margin, y);
      ctx.lineTo(width - margin, y);
    }
    ctx.stroke();

    // Draw axis lines.
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, height - margin);
    ctx.lineTo(width - margin, height - margin);
    ctx.stroke();

    // Draw loss curve (red).
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((d, i) => {
      const x = margin + (d.epoch / maxEpoch) * chartWidth;
      const y = height - margin - (d.loss / maxLoss) * chartHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw accuracy curve (blue), scaled to [0,100].
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((d, i) => {
      const x = margin + (d.epoch / maxEpoch) * chartWidth;
      const y = height - margin - (d.accuracy / 100) * chartHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw legends.
    ctx.fillStyle = 'red';
    ctx.font = '12px sans-serif';
    ctx.fillText('Loss', margin, margin - 10);
    ctx.fillStyle = 'blue';
    ctx.fillText('Accuracy', margin + 50, margin - 10);
  }, []);

  // Redraw chart whenever trainingHistory updates.
  useEffect(() => {
    drawTrainingChart(trainingHistory);
  }, [trainingHistory, drawTrainingChart]);
  // ------------------------------

  // Add mouse events for chart tooltip.
  useEffect(() => {
    const canvas = chartCanvasRef.current;
    if (!canvas) return;
    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const xPos = e.clientX - rect.left;
      const margin = 30;
      const chartWidth = canvas.width - 2 * margin;
      // Map x position to an approximate epoch.
      const approxEpoch = ((xPos - margin) / chartWidth) * (trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].epoch : 1);
      let nearest = null;
      let minDiff = Infinity;
      trainingHistory.forEach((d) => {
        const diff = Math.abs(d.epoch - approxEpoch);
        if (diff < minDiff) {
          minDiff = diff;
          nearest = d;
        }
      });
      if (nearest) {
        setChartTooltip({
          x: e.clientX,
          y: rect.top + margin,
          data: nearest,
        });
      }
    };
    const handleMouseLeave = () => {
      setChartTooltip(null);
    };
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);
    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [trainingHistory]);

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
    setTrainingHistory([]);
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
    const layers = [...hiddenLayerSizes, 1];
    modelRef.current = new MLP(2, layers);
    updateModelInsights();
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  // Compute and update model insights.
  const updateModelInsights = () => {
    if (modelRef.current) {
      const params = modelRef.current.parameters();
      let total = 0;
      params.forEach(() => total += 1);
      setModelInsights({
        numLayers: params.length,
        totalParams: total,
      });
    } else {
      setModelInsights(null);
    }
  };

  // Compute summary of model parameters per layer.
  const updateModelSummary = () => {
    if (!modelRef.current) return;
    // Get layers from the MLP instance.
    const layers = modelRef.current.layers;
    const summaries = layers.map((layer, index) => {
      // Each layer is a Layer with neurons.
      const neuronSummaries = layer.neurons.map(neuron => {
        const weights = neuron.w.map(w => w.data);
        const avgWeight = weights.reduce((sum, w) => sum + w, 0) / weights.length;
        return { avgWeight, bias: neuron.b.data };
      });
      const avgLayerWeight = neuronSummaries.reduce((sum, n) => sum + n.avgWeight, 0) / neuronSummaries.length;
      const avgLayerBias = neuronSummaries.reduce((sum, n) => sum + n.bias, 0) / neuronSummaries.length;
      return {
        layerName: index === layers.length - 1 ? 'Output Layer' : `Hidden Layer ${index + 1}`,
        avgWeight: avgLayerWeight,
        avgBias: avgLayerBias,
        activation: index === layers.length - 1 ? 'Sigmoid' : 'ReLU',
      };
    });
    // Add an input layer summary.
    const inputSummary = { layerName: 'Input Layer', neurons: 2, label: 'Input [x, y]' };
    setModelSummary([inputSummary, ...summaries]);
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
    setTrainingHistory([]);
    trainModel();
  };

  const stopTraining = () => {
    setIsTraining(false);
    isTrainingRef.current = false;
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  // Export training history as CSV.
  const exportTrainingData = () => {
    let csv = 'Epoch,Loss,Accuracy\n';
    trainingHistory.forEach((d) => {
      csv += `${d.epoch},${d.loss},${d.accuracy}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'training_history.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Clear training history.
  const clearTrainingHistory = () => {
    setTrainingHistory([]);
  };

  // Compute decision boundary grid (lower resolution to speed drawing).
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
    const currentBatchSize = batchSizeRef.current;

    const start = batchIndexRef.current;
    const end = Math.min(start + currentBatchSize, totalSamples);
    let miniBatchLoss = new Value(0);
    let miniBatchCorrect = 0;
    for (let i = start; i < end; i++) {
      const [x, y] = points[i];
      const target = labels[i]; // 0 or 1
      const rawOutput = model.call([x, y])[0];
      const p = sigmoid(rawOutput);
      const pred = p.data > 0.5 ? 1 : 0;
      if (pred === target) miniBatchCorrect++;
      const diff = p.add(new Value(-target));
      const sampleLoss = diff.mul(diff);
      miniBatchLoss = miniBatchLoss.add(sampleLoss);
    }
    miniBatchLoss.backward();
    const params = model.parameters();
    for (const p of params) {
      p.data -= learningRate * p.grad;
    }
    model.zero_grad();

    const batchCount = end - start;
    epochLossRef.current += miniBatchLoss.data;
    epochCorrectRef.current += miniBatchCorrect;
    epochCountRef.current += batchCount;
    batchIndexRef.current = end;

    if (batchIndexRef.current >= totalSamples) {
      const avgLoss = epochLossRef.current / epochCountRef.current;
      const acc = (epochCorrectRef.current / epochCountRef.current) * 100;
      setLoss(avgLoss);
      setAccuracy(acc);
      setEpochs((prev) => prev + 1);
      // Use training history length for current epoch.
      setTrainingHistory((prev) => {
        const currentEpoch = prev.length + 1;
        return [...prev, { epoch: currentEpoch, loss: avgLoss, accuracy: acc }];
      });
      updateModelSummary();

      const grid = generateModelData();
      drawVisualization(points, labels, grid);

      if (enableLrDecay) {
        setLearningRate((prev) => prev * 0.98);
      }

      if (acc >= 98) {
        stopTraining();
        console.log('Target accuracy reached:', acc);
        return;
      }
      batchIndexRef.current = 0;
      epochLossRef.current = 0;
      epochCorrectRef.current = 0;
      epochCountRef.current = 0;
    }

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

    if (grid && grid.length > 0) {
      for (const point of grid) {
        const x = translateX + point.x * scale;
        const y = translateY - point.y * scale;
        ctx.fillStyle = interpolateColor(point.value * 2 - 1);
        ctx.fillRect(x - scale / 2, y - scale / 2, scale, scale);
      }
    }
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
        <p className="text-gray-600">
          Watch a neural network learn to classify data in real-time with advanced training options
        </p>
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
                  onChange={(e) =>
                    updateHiddenLayers(e.target.value.split(',').map(Number))
                  }
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
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              onClick={exportTrainingData}
              className="px-4 py-2 bg-green-600 text-white rounded"
            >
              Export Training Data
            </button>
            <button
              onClick={clearTrainingHistory}
              className="px-4 py-2 bg-yellow-600 text-white rounded"
            >
              Clear Training History
            </button>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow p-4 space-y-4">
          <div>
            <h2 className="text-xl font-bold mb-2">Network Architecture</h2>
            {/* <NetworkArchitectureVisualizer
              inputNeurons={2}
              hiddenLayers={hiddenLayerSizes}
              outputNeurons={1}
              outputActivation="Sigmoid"
              hiddenActivation="ReLU"
            /> */}
            <NetworkArchitectureVisualizer layersSummary={modelSummary} />
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
            <h2 className="text-xl font-bold mb-2">Advanced Training Options</h2>
            <div className="space-y-2">
              <div className="flex items-center">
                <label className="text-sm font-medium mr-2">Batch Size:</label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  disabled={isTrainingRef.current}
                />
                <span className="text-sm ml-2">{batchSize}</span>
              </div>
              <div className="flex items-center">
                <label className="text-sm font-medium mr-2">Learning Rate Decay:</label>
                <input
                  type="checkbox"
                  checked={enableLrDecay}
                  onChange={(e) => setEnableLrDecay(e.target.checked)}
                  disabled={isTrainingRef.current}
                />
              </div>
            </div>
          </div>
          <div>
            <h2 className="text-xl font-bold mb-2">Advanced Model Insights</h2>
            {modelInsights ? (
              <div className="text-sm">
                <p>Number of Layers: {modelInsights.numLayers}</p>
                <p>Total Parameters: {modelInsights.totalParams}</p>
              </div>
            ) : (
              <p className="text-sm text-gray-500">No model insights available.</p>
            )}
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
          <div className="relative">
            <h2 className="text-xl font-bold mb-2">Training Chart</h2>
            <canvas
              ref={chartCanvasRef}
              width={600}
              height={300}
              className="border border-gray-200 rounded"
              style={{ backgroundColor: '#fff' }}
            />
            {chartTooltip && (
              <div
                style={{
                  position: 'absolute',
                  left: chartTooltip.x - 50,
                  top: chartTooltip.y - 40,
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  color: '#fff',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  pointerEvents: 'none',
                  fontSize: '12px'
                }}
              >
                <div>Epoch: {chartTooltip.data.epoch}</div>
                <div>Loss: {chartTooltip.data.loss.toFixed(4)}</div>
                <div>Acc: {chartTooltip.data.accuracy.toFixed(2)}%</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NeuralNetworkVisualizer;
