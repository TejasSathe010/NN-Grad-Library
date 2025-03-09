import React from 'react';

const PackagePage = () => {
  return (
    <div className="bg-white dark:bg-gray-800 text-gray-900 dark:text-white font-sans">
      <div className="max-w-3xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 dark:text-white leading-tight">NN-Grad Library</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400 mt-4">
            Neural Network Training and Visualization Library for JavaScript/React. 
            Use this library to train and visualize neural networks easily.
          </p>
        </div>

        <div className="mb-10">
          <h2 className="text-3xl font-semibold text-gray-800 dark:text-white">Installation</h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 mt-2">You can install and use `nn-grad` in two ways:</p>

          <div className="mt-6">
            <h3 className="text-2xl font-semibold text-gray-800 dark:text-white">Install with NPM</h3>
            <p className="text-gray-600 dark:text-gray-400 mt-2">To install `nn-grad` via npm, run the following command:</p>
            <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
              <code>npm install nn-grad</code>
            </pre>
          </div>

          {/* Git Clone and Setup */}
          <div className="mt-6">
            <h3 className="text-2xl font-semibold text-gray-800 dark:text-white">Install via Git Clone</h3>
            <p className="text-gray-600 dark:text-gray-400 mt-2">Alternatively, you can clone the repository and link it locally:</p>
            <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
              <code>
                git clone git@github.com:TejasSathe010/NN-Grad-Library.git
                <br />
                cd NN-Grad-Library
                <br />
                npm install
                <br />
                npm link
              </code>
            </pre>
            <p className="text-gray-600 dark:text-gray-400 mt-2">After setting up the library locally, you can use it in your React project:</p>
            <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
              <code>npm link nn-grad</code>
            </pre>
          </div>
        </div>

        <div className="mb-10">
          <h2 className="text-3xl font-semibold text-gray-800 dark:text-white">Usage</h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 mt-2">You can use the library as follows:</p>
          <div className="mt-4">
            <h3 className="text-2xl font-semibold text-gray-800 dark:text-white">Import the Library</h3>
            <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
              <code>import &#123; MLP, Value &#125; from 'nngrad';</code>
            </pre>

            <h3 className="text-2xl font-semibold text-gray-800 dark:text-white mt-6">Creating and Using Models</h3>
            <p className="text-gray-600 dark:text-gray-400 mt-2">Here's an example of how to create a model and use the `MLP` (Multi-Layer Perceptron) and `Value` classes:</p>
            <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
              <code>
                const layers = &#91;10, 5, 2&#93;;
                modelRef.current = new MLP(2, layers);
              </code>
            </pre>
            <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
              <code>
                const out = new Value(s, &#91;x&#93;, 'sigmoid');
                </code>
            </pre>
            <pre className="bg-gray-800 text-white p-4 rounded-md mt-2">
              <code>
                let miniBatchLoss = new Value(0);
              </code>
            </pre>
            <p className="text-gray-600 dark:text-gray-400 mt-2">In this example, we use `MLP` to create a model with two input features and three layers. Then we define a `Value` to represent the output of the network and calculate loss.</p>
          </div>
        </div>

        <div className="mt-12 text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            © 2025 Tejas Sathe. All rights reserved. 
            <a href="https://github.com/TejasSathe010/NN-Grad-Library" className="text-blue-500 hover:underline">GitHub Repository</a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default PackagePage;
