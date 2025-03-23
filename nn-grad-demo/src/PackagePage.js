import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

const PackagePage = () => {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  const opacity = useTransform(scrollYProgress, [0, 0.3], [0.3, 1]);
  const y = useTransform(scrollYProgress, [0, 0.3], [50, 0]);

  return (
    <motion.div
      ref={containerRef}
      style={{ opacity, y }}
      className="min-h-screen bg-gray-50 dark:bg-gray-900"
    >
      <div className="max-w-3xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-16 text-center"
        >
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl md:text-5xl font-extrabold text-gray-900 dark:text-white leading-tight mb-4"
          >
            NN-Grad Library
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg md:text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto"
          >
            Neural Network Training and Visualization Library for JavaScript/React. 
            Use this library to train and visualize neural networks easily.
          </motion.p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden ring-1 ring-black ring-opacity-5 mb-12"
        >
          <div className="px-6 py-8">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-3xl font-semibold text-gray-900 dark:text-white mb-6"
            >
              Installation
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-lg text-gray-600 dark:text-gray-400 mb-8"
            >
              You can install and use `nn-grad` in two ways:
            </motion.p>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <h3 className="text-2xl font-semibold text-gray-800 dark:text-white mb-4">
                Install with NPM
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                To install `nn-grad` via npm, run the following command:
              </p>
              <div className="bg-gray-800 text-white p-6 rounded-xl mb-8">
                <code className="text-lg">npm install nn-grad</code>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <h3 className="text-2xl font-semibold text-gray-800 dark:text-white mb-4">
                Install via Git Clone
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Alternatively, you can clone the repository and link it locally:
              </p>
              <div className="bg-gray-800 text-white p-6 rounded-xl mb-4">
                <code className="text-lg">
                  git clone git@github.com:TejasSathe010/NN-Grad-Library.git<br />
                  cd NN-Grad-Library<br />
                  npm install<br />
                  npm link
                </code>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                After setting up the library locally, you can use it in your React project:
              </p>
              <div className="bg-gray-800 text-white p-6 rounded-xl">
                <code className="text-lg">npm link nn-grad</code>
              </div>
            </motion.div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden ring-1 ring-black ring-opacity-5 mb-12"
        >
          <div className="px-6 py-8">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-3xl font-semibold text-gray-900 dark:text-white mb-6"
            >
              Usage
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-lg text-gray-600 dark:text-gray-400 mb-8"
            >
              You can use the library as follows:
            </motion.p>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <h3 className="text-2xl font-semibold text-gray-800 dark:text-white mb-4">
                Import the Library
              </h3>
              <div className="bg-gray-800 text-white p-6 rounded-xl mb-8">
                <code className="text-lg">{"import { MLP, Value } from 'nngrad';"}</code>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <h3 className="text-2xl font-semibold text-gray-800 dark:text-white mb-4">
                Creating and Using Models
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Here's an example of how to create a model and use the `MLP` (Multi-Layer Perceptron) and `Value` classes:
              </p>
              <div className="bg-gray-800 text-white p-6 rounded-xl mb-4">
                <code className="text-lg">
                  const layers = [10, 5, 2];<br />
                  modelRef.current = new MLP(2, layers);
                </code>
              </div>
              <div className="bg-gray-800 text-white p-6 rounded-xl mb-4">
                <code className="text-lg">
                  const out = new Value(s, [x], 'sigmoid');
                </code>
              </div>
              <div className="bg-gray-800 text-white p-6 rounded-xl">
                <code className="text-lg">
                  let miniBatchLoss = new Value(0);
                </code>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mt-4">
                In this example, we use `MLP` to create a model with two input features and three layers. Then we define a `Value` to represent the output of the network and calculate loss.
              </p>
            </motion.div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="text-center pb-12"
        >
          <p className="text-sm text-gray-500 dark:text-gray-500 mb-2">
            © 2025 Tejas Sathe. All rights reserved.
          </p>
          <a
            href="https://github.com/TejasSathe010/NN-Grad-Library"
            className="text-blue-600 dark:text-blue-400 hover:underline text-sm"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub Repository
          </a>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default PackagePage;