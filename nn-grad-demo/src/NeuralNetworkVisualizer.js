import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import ReactFlow, { Controls, Background } from 'react-flow-renderer';
import { MLP, Value } from 'nngrad';

const sigmoid = (x) => {
  const s = 1 / (1 + Math.exp(-x.data));
  const out = new Value(s, [x], 'sigmoid');
  out._backward = function () {
    x.grad += out.data * (1 - out.data) * out.grad;
  };
  return out;
};

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

  useEffect(() => {
    isTrainingRef.current = isTraining;
  }, [isTraining]);

  useEffect(() => {
    batchSizeRef.current = batchSize;
  }, [batchSize]);

  const drawTrainingChart = useCallback((history) => {
    if (!chartCanvasRef.current) return;
    const canvas = chartCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);

    const margin = 30;
    const chartWidth = width - 2 * margin;
    const chartHeight = height - 2 * margin;

    const maxEpoch = history.length > 0 ? history[history.length - 1].epoch : 1;
    const losses = history.map((d) => d.loss);
    const maxLoss = losses.length > 0 ? Math.max(...losses) : 1;

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

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, height - margin);
    ctx.lineTo(width - margin, height - margin);
    ctx.stroke();

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

    ctx.fillStyle = 'red';
    ctx.font = '12px sans-serif';
    ctx.fillText('Loss', margin, margin - 10);
    ctx.fillStyle = 'blue';
    ctx.fillText('Accuracy', margin + 50, margin - 10);
  }, []);

  useEffect(() => {
    drawTrainingChart(trainingHistory);
  }, [trainingHistory, drawTrainingChart]);
  // ------------------------------

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
    batchIndexRef.current = 0;
    epochLossRef.current = 0;
    epochCorrectRef.current = 0;
    epochCountRef.current = 0;
    initializeModel();
    const grid = generateModelData();
    drawVisualization(data.points, data.labels, grid);
  };

  const initializeModel = () => {
    const layers = [...hiddenLayerSizes, 1];
    modelRef.current = new MLP(2, layers);
    updateModelInsights();
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

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

  const updateModelSummary = () => {
    if (!modelRef.current) return;
    const layers = modelRef.current.layers;
    const summaries = layers.map((layer, index) => {
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
    const inputSummary = { layerName: 'Input Layer', neurons: 2, label: 'Input [x, y]' };
    setModelSummary([inputSummary, ...summaries]);
  };

  const startTraining = () => {
    if (isTrainingRef.current) return;
    setIsTraining(true);
    isTrainingRef.current = true;
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

  const clearTrainingHistory = () => {
    setTrainingHistory([]);
  };

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

  const updateHiddenLayers = (config) => {
    setHiddenLayerSizes(config);
    initializeModel();
    setEpochs(0);
    setLoss(0);
    setAccuracy(0);
  };

  return (
    <div className="w-full bg-white dark:bg-slate-900 font-sans">
  <div className="max-w-8xl mx-auto px-4 sm:px-6 md:px-8">
    <div className="py-8 mb-6 border-b border-slate-200 dark:border-slate-700">
  <div className="flex items-center justify-between">
    <div>
      <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-900 dark:text-white tracking-tight">
        Neural Network Training Visualizer
      </h1>
      <p className="mt-3 text-lg text-slate-600 dark:text-slate-400 max-w-3xl">
        Watch a neural network learn to classify data in real-time with advanced training options
      </p>
    </div>
    <div className="hidden lg:block">
    <Link
        to="/package-page" 
        className="inline-flex items-center rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700 focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 dark:bg-sky-500 dark:hover:bg-sky-400"
      >
        Checkout nngrad library
        <svg className="ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
        </svg>
      </Link>
    </div>
  </div>
</div>

    <div className="lg:flex lg:items-start">
      <div className="flex-1 py-10">
        <div className="mb-10 overflow-hidden rounded-lg ring-1 ring-slate-200 dark:ring-slate-800 shadow-sm">
          <div className="p-6">
            <canvas
              ref={canvasRef}
              className="w-full h-96 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
              width={800}
              height={400}
            />
            
            <div className="mt-8 flex flex-wrap items-center gap-4">
              <button
                onClick={startTraining}
                disabled={isTrainingRef.current}
                className="inline-flex items-center justify-center rounded-md py-2 px-4 text-sm font-semibold focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 bg-sky-500 text-white hover:bg-sky-600 active:bg-sky-700 focus-visible:outline-sky-600 disabled:bg-slate-300 disabled:text-slate-500"
              >
                Start Training
              </button>
              <button
                onClick={stopTraining}
                disabled={!isTrainingRef.current}
                className="inline-flex items-center justify-center rounded-md py-2 px-4 text-sm font-semibold focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 bg-rose-500 text-white hover:bg-rose-600 active:bg-rose-700 focus-visible:outline-rose-600 disabled:bg-slate-300 disabled:text-slate-500"
              >
                Stop Training
              </button>
              <button
                onClick={generateData}
                disabled={isTrainingRef.current}
                className="inline-flex items-center justify-center rounded-md py-2 px-4 text-sm font-semibold focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 bg-slate-800 text-white hover:bg-slate-900 active:bg-slate-800 focus-visible:outline-slate-900 disabled:bg-slate-300 disabled:text-slate-500 dark:bg-slate-700 dark:hover:bg-slate-600"
              >
                Regenerate Data
              </button>
            </div>
            
            <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Dataset</label>
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  disabled={isTrainingRef.current}
                  className="block w-full rounded-md border-slate-300 dark:border-slate-700 text-sm shadow-sm py-2 pl-3 pr-10 text-slate-900 dark:text-white bg-white dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:bg-slate-100 dark:disabled:bg-slate-800 disabled:cursor-not-allowed"
                >
                  <option value="spiral">Spiral</option>
                  <option value="circle">Circle</option>
                  <option value="moon">Moon</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Learning Rate</label>
                <select
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  disabled={isTrainingRef.current}
                  className="block w-full rounded-md border-slate-300 dark:border-slate-700 text-sm shadow-sm py-2 pl-3 pr-10 text-slate-900 dark:text-white bg-white dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:bg-slate-100 dark:disabled:bg-slate-800 disabled:cursor-not-allowed"
                >
                  <option value="0.001">0.001</option>
                  <option value="0.005">0.005</option>
                  <option value="0.01">0.01</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">Architecture</label>
                <select
                  value={hiddenLayerSizes.join(',')}
                  onChange={(e) =>
                    updateHiddenLayers(e.target.value.split(',').map(Number))
                  }
                  disabled={isTrainingRef.current}
                  className="block w-full rounded-md border-slate-300 dark:border-slate-700 text-sm shadow-sm py-2 pl-3 pr-10 text-slate-900 dark:text-white bg-white dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500 disabled:bg-slate-100 dark:disabled:bg-slate-800 disabled:cursor-not-allowed"
                >
                  <option value="5">5 neurons</option>
                  <option value="10">10 neurons</option>
                  <option value="5,5">5,5 neurons</option>
                  <option value="10,10">10,10 neurons</option>
                  <option value="20,10,5">20,10,5 neurons</option>
                </select>
              </div>
            </div>
            
            <div className="mt-8 flex flex-wrap gap-4">
              <button
                onClick={exportTrainingData}
                className="inline-flex items-center justify-center rounded-md py-2 px-4 text-sm font-semibold focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 bg-emerald-500 text-white hover:bg-emerald-600 active:bg-emerald-700 focus-visible:outline-emerald-600"
              >
                Export Training Data
              </button>
              <button
                onClick={clearTrainingHistory}
                className="inline-flex items-center justify-center rounded-md py-2 px-4 text-sm font-semibold focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-2 bg-amber-500 text-white hover:bg-amber-600 active:bg-amber-700 focus-visible:outline-amber-600"
              >
                Clear Training History
              </button>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-10">
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white">Training Stats</h2>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800">
              <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800 text-sm font-medium text-slate-900 dark:text-slate-200">
                Current metrics
              </div>
              <div className="divide-y divide-slate-200 dark:divide-slate-800">
                <div className="flex justify-between px-4 py-3 text-sm">
                  <span className="text-slate-600 dark:text-slate-400">Epochs</span>
                  <span className="font-medium text-slate-900 dark:text-white">{epochs}</span>
                </div>
                <div className="flex justify-between px-4 py-3 text-sm">
                  <span className="text-slate-600 dark:text-slate-400">Loss</span>
                  <span className="font-medium text-slate-900 dark:text-white">{loss.toFixed(4)}</span>
                </div>
                <div className="flex justify-between px-4 py-3 text-sm">
                  <span className="text-slate-600 dark:text-slate-400">Accuracy</span>
                  <span className="font-medium text-slate-900 dark:text-white">{accuracy.toFixed(2)}%</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white">Advanced Training Options</h2>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800">
              <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800 text-sm font-medium text-slate-900 dark:text-slate-200">
                Hyperparameters
              </div>
              <div className="divide-y divide-slate-200 dark:divide-slate-800">
                <div className="flex items-center justify-between px-4 py-3">
                  <label className="text-sm text-slate-600 dark:text-slate-400">Batch Size</label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="5"
                      max="50"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value))}
                      disabled={isTrainingRef.current}
                      className="w-24 accent-sky-500"
                    />
                    <span className="inline-flex items-center rounded-md bg-slate-50 dark:bg-slate-800 px-2 py-1 text-xs font-medium text-slate-600 dark:text-slate-400 ring-1 ring-inset ring-slate-500/10 dark:ring-slate-400/20">
                      {batchSize}
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between px-4 py-3">
                  <label className="text-sm text-slate-600 dark:text-slate-400">Learning Rate Decay</label>
                  <div className="relative flex items-start">
                    <div className="flex h-6 items-center">
                      <input
                        type="checkbox"
                        checked={enableLrDecay}
                        onChange={(e) => setEnableLrDecay(e.target.checked)}
                        disabled={isTrainingRef.current}
                        className="h-4 w-4 rounded border-slate-300 text-sky-600 focus:ring-sky-600 disabled:opacity-50"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-10">
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white">Model Insights</h2>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800">
              <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800 text-sm font-medium text-slate-900 dark:text-slate-200">
                Architecture details
              </div>
              <div className="divide-y divide-slate-200 dark:divide-slate-800">
                {modelInsights ? (
                  <>
                    <div className="flex justify-between px-4 py-3 text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Layers</span>
                      <span className="font-medium text-slate-900 dark:text-white">{modelInsights.numLayers}</span>
                    </div>
                    <div className="flex justify-between px-4 py-3 text-sm">
                      <span className="text-slate-600 dark:text-slate-400">Parameters</span>
                      <span className="font-medium text-slate-900 dark:text-white">{modelInsights.totalParams}</span>
                    </div>
                  </>
                ) : (
                  <div className="px-4 py-3 text-sm italic text-slate-500 dark:text-slate-400">
                    No model insights available.
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white">Legend</h2>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800">
              <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800 text-sm font-medium text-slate-900 dark:text-slate-200">
                Data visualization
              </div>
              <div className="divide-y divide-slate-200 dark:divide-slate-800">
                <div className="flex items-center gap-3 px-4 py-3">
                  <div className="w-4 h-4 rounded-full bg-red-500"></div>
                  <span className="text-sm text-slate-600 dark:text-slate-400">Class 0</span>
                </div>
                <div className="flex items-center gap-3 px-4 py-3">
                  <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                  <span className="text-sm text-slate-600 dark:text-slate-400">Class 1</span>
                </div>
                <div className="flex items-center gap-3 px-4 py-3">
                  <div className="w-16 h-4 bg-gradient-to-r from-red-500 to-blue-500 rounded"></div>
                  <span className="text-sm text-slate-600 dark:text-slate-400">Decision Boundary</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="not-prose relative bg-slate-50 rounded-xl overflow-hidden dark:bg-slate-800/25">
        <div className="absolute inset-0 bg-grid-slate-100 [mask-image:linear-gradient(0deg,#fff,rgba(255,255,255,0.6))] dark:bg-grid-slate-700/25 dark:[mask-image:linear-gradient(0deg,rgba(255,255,255,0.1),rgba(255,255,255,0.5))]"></div>
        <div className="relative rounded-xl overflow-auto">
          <div className="my-8 space-y-6 relative">
            <div className="px-4">
              <h2 className="text-base font-semibold text-slate-900 dark:text-slate-200 flex items-center space-x-2">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-sky-500 dark:text-sky-400">
                  <path d="M6 6H8V18H6V6Z" fill="currentColor" />
                  <path d="M16 6H18V18H16V6Z" fill="currentColor" />
                  <path d="M21 18H3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  <path d="M3 15.5005L16.443 8.12325C16.8035 7.93591 17.2264 8.0601 17.4138 8.42063C17.4925 8.56191 17.511 8.72856 17.4653 8.88282L15.1258 16.0903C15.0435 16.3786 14.7733 16.5729 14.4748 16.5376C14.3372 16.5221 14.2063 16.4611 14.1029 16.3636L9.63615 12.1589C9.53862 12.0671 9.40969 12.0129 9.27411 12.0064L3 11.5005" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
                <span>Training Progress</span>
              </h2>
              <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">Real-time visualization of model performance metrics during training</p>
            </div>
            
            <div className="relative px-4">
              <div className="absolute inset-0 -mx-4 border-t border-slate-200 dark:border-slate-700"></div>
              <div className="relative">
                <div className="flex items-center justify-between pt-4 pb-2">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-sky-500"></div>
                    <span className="text-xs font-medium text-slate-600 dark:text-slate-400">Loss</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-indigo-500"></div>
                    <span className="text-xs font-medium text-slate-600 dark:text-slate-400">Accuracy</span>
                  </div>
                  <div className="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
                    <span>Total Epochs: <span className="font-semibold text-slate-700 dark:text-slate-300">{epochs}</span></span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="rounded-lg shadow-sm bg-white dark:bg-slate-800 overflow-hidden ring-1 ring-slate-900/5 dark:ring-white/10">
              <div className="p-4 md:p-6">
                <canvas
                  ref={chartCanvasRef}
                  width={900}
                  height={400}
                  className="w-full h-full"
                />
              </div>
              <div className="border-t border-slate-200 dark:border-slate-700 px-4 py-3 sm:px-6 flex justify-between items-center bg-slate-50 dark:bg-slate-800/60">
                <button className="inline-flex items-center text-xs font-medium text-slate-700 dark:text-slate-300 hover:text-sky-500 dark:hover:text-sky-400">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                  </svg>
                  Export as PNG
                </button>
                <div className="flex items-center">
                  <span className="text-xs text-slate-600 dark:text-slate-400 mr-3">
                    Current Loss: <span className="font-mono font-medium text-slate-900 dark:text-white">{(loss || 0).toFixed(4)}</span>
                  </span>
                  <span className="text-xs text-slate-600 dark:text-slate-400">
                    Accuracy: <span className="font-mono font-medium text-slate-900 dark:text-white">{(accuracy || 0).toFixed(2)}%</span>
                  </span>
                </div>
              </div>
            </div>
            
            {chartTooltip && (
              <div className="absolute z-10 px-3 py-2 text-sm font-medium text-white bg-slate-800 rounded-lg shadow-sm dark:bg-slate-700 pointer-events-none transform -translate-x-1/2 -translate-y-full"
                style={{
                  left: chartTooltip.x,
                  top: chartTooltip.y - 8,
                }}
              >
                <div className="flex items-center justify-between space-x-4 mb-1 pb-1 border-b border-slate-700/50">
                  <span className="font-medium text-sky-400">Epoch {chartTooltip.data.epoch}</span>
                </div>
                <div className="flex flex-col space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">Loss:</span>
                    <span className="font-mono font-medium text-sky-300">{chartTooltip.data.loss.toFixed(4)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">Accuracy:</span>
                    <span className="font-mono font-medium text-indigo-300">{chartTooltip.data.accuracy.toFixed(2)}%</span>
                  </div>
                </div>
                <div className="absolute left-1/2 top-full -translate-x-1/2 -mt-px border-8 border-transparent border-t-slate-800 dark:border-t-slate-700"></div>
              </div>
            )}
          </div>
        </div>
        <div className="absolute inset-0 pointer-events-none border border-slate-900/5 rounded-xl dark:border-slate-100/5"></div>
        </div>

        <div className="mb-10 space-y-6">
          <h2 className="text-2xl font-semibold text-gray-900">Network Architecture</h2>
          <NetworkArchitectureVisualizer layersSummary={modelSummary} />
        </div>

      </div>
    </div>
  </div>
</div>
  );
};

export default NeuralNetworkVisualizer;
