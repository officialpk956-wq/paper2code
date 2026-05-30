"use client";
import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';

// Dynamic import for react-plotly.js to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function MathVisualization() {
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    // Generate a simple gradient descent visualization data point
    const x = [];
    const y = [];
    for (let i = -5; i <= 5; i += 0.5) {
      x.push(i);
      y.push(i * i * learningRate);
    }
    setData([
      {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: 'cyan' },
      }
    ]);
  }, [learningRate]);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 rounded-2xl bg-gray-900/50 backdrop-blur border border-white/10"
    >
      <h3 className="text-xl font-bold text-white mb-4">Gradient Visualization</h3>
      
      <div className="mb-6">
        <label className="text-sm text-gray-300 block mb-2">Learning Rate: {learningRate}</label>
        <input 
          type="range" 
          min="0.001" 
          max="0.1" 
          step="0.001" 
          value={learningRate} 
          onChange={(e) => setLearningRate(parseFloat(e.target.value))}
          className="w-full accent-cyan-500"
        />
      </div>

      <div className="w-full h-64 bg-black/40 rounded-lg overflow-hidden">
        <Plot
          data={data}
          layout={{
            autosize: true,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { t: 10, b: 20, l: 30, r: 10 },
            xaxis: { color: '#666', gridcolor: '#333' },
            yaxis: { color: '#666', gridcolor: '#333' }
          }}
          useResizeHandler={true}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    </motion.div>
  );
}
