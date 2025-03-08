# NN-Grad Library

**NN-Grad** is a Neural Network Training and Visualization Library for JavaScript/React. It provides a simple API to create, train, and visualize neural networks in a React environment. With easy-to-use classes such as `MLP` (Multi-Layer Perceptron) and `Value`, this library enables rapid experimentation with machine learning models in JavaScript.

## Features

- Create neural networks with custom architectures using `MLP`.
- Visualize the training process with interactive graphs and charts.
- Easy integration with React for real-time updates.
- Lightweight, no dependencies beyond core JavaScript and React.

## Installation

You can install and use `nn-grad` in two ways:

### 1. Install with NPM

To install `nn-grad` via npm, run the following command:

```bash
npm install nn-grad
```

### 2. Install via Git Clone

Alternatively, you can clone the repository and set it up locally:

```bash
git clone git@github.com:TejasSathe010/NN-Grad-Library.git
cd NN-Grad-Library
npm install
npm link
```

After setting up the library locally, link it to your project:

```bash
npm link nn-grad
```

## Usage

You can start using the library by importing the necessary classes and initializing a model.

### Import the Library

```js
import { MLP, Value } from 'nngrad';
```

### Creating and Using Models

Here's an example of how to create a model and use the `MLP` (Multi-Layer Perceptron) and `Value` classes:

```js
const layers = [10, 5, 2];
modelRef.current = new MLP(2, layers);

const out = new Value(s, [x], 'sigmoid');
let miniBatchLoss = new Value(0);
```

In this example:
- We use `MLP` to create a model with two input features and three layers.
- We define a `Value` to represent the output of the network and calculate the loss.

### Example of Training Process

You can implement a simple training loop and visualize the training progress. Below is a basic implementation for training with real-time accuracy and loss updates:

```js
const trainModel = async () => {
  // Training loop for model
  for (let epoch = 0; epoch < 1000; epoch++) {
    modelRef.current.train(inputs, targets, learningRate);
    const loss = modelRef.current.computeLoss();
    const accuracy = modelRef.current.computeAccuracy();
    
    // Update loss and accuracy in UI
    setLoss(loss);
    setAccuracy(accuracy);

    if (accuracy >= 98) {
      break; // Stop training when accuracy is above 98%
    }
  }
};
```

This will stop training once the accuracy reaches 98%.

## Development

To contribute to the project or make modifications locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone git@github.com:TejasSathe010/NN-Grad-Library.git
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Link the library:
   ```bash
   npm link
   ```

4. Make your changes and test locally.

5. To test the library in your React project, you can link it by running:
   ```bash
   npm link nn-grad
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

© 2025 Tejas Sathe. All rights reserved. [GitHub Repository](https://github.com/TejasSathe010/NN-Grad-Library)
```
