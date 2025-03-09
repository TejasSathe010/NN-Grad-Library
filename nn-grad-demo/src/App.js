import NeuralNetworkVisualizer from './NeuralNetworkVisualizer';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import PackagePage from './PackagePage';


function App() {
  return (
    <div className="App">
      <Router>
        <Routes>
          <Route exact path="/" element={<NeuralNetworkVisualizer />} />
          <Route path="/package-page" element={<PackagePage />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
