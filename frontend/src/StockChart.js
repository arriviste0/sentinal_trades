import React, { useEffect, useState } from 'react';
import { predictPrice } from './api';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceDot
} from 'recharts';

// Sample historical price data (replace with real data or fetch from backend)
const SAMPLE_PRICES = [
  { date: '2024-07-05', open: 150, high: 153, low: 149, close: 152, volume: 12000000, avg_sentiment_score: 0.2, news_count: 3 },
  { date: '2024-07-06', open: 152, high: 154, low: 151, close: 153, volume: 11000000, avg_sentiment_score: 0.1, news_count: 2 },
  { date: '2024-07-07', open: 153, high: 155, low: 152, close: 154, volume: 11500000, avg_sentiment_score: 0.3, news_count: 4 },
  { date: '2024-07-08', open: 154, high: 156, low: 153, close: 155, volume: 13000000, avg_sentiment_score: 0.4, news_count: 5 },
  { date: '2024-07-09', open: 155, high: 157, low: 154, close: 156, volume: 12500000, avg_sentiment_score: 0.5, news_count: 6 },
];

function StockChart() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchPrediction() {
      setLoading(true);
      setError('');
      try {
        // Prepare the last 5 days for prediction
        const data = SAMPLE_PRICES.map(day => ({
          open: day.open,
          high: day.high,
          low: day.low,
          close: day.close,
          volume: day.volume,
          avg_sentiment_score: day.avg_sentiment_score,
          news_count: day.news_count,
        }));
        const result = await predictPrice(data);
        setPrediction(result.predicted_pct_change);
      } catch (err) {
        setError('Error fetching price prediction.');
      } finally {
        setLoading(false);
      }
    }
    fetchPrediction();
  }, []);

  // Calculate predicted next close price
  const lastClose = SAMPLE_PRICES[SAMPLE_PRICES.length - 1].close;
  const predictedClose = prediction !== null ? lastClose * (1 + prediction) : null;
  const chartData = [...SAMPLE_PRICES.map(row => ({ ...row })),
    predictedClose !== null ? {
      date: 'Predicted',
      close: predictedClose,
      predicted: true
    } : null
  ].filter(Boolean);

  return (
    <div className="stock-chart">
      <h2>Stock Price Chart</h2>
      {loading && <div className="chart-placeholder">Loading chart...</div>}
      {error && <div className="chart-placeholder" style={{color: 'red'}}>{error}</div>}
      {!loading && !error && (
        <>
          <div style={{ width: '100%', height: 300, marginBottom: 16 }}>
            <ResponsiveContainer>
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Line type="monotone" dataKey="close" stroke="#8884d8" dot={!predictedClose} />
                {predictedClose !== null && (
                  <ReferenceDot x="Predicted" y={predictedClose} r={8} fill="#82ca9d" stroke="none" label={{ value: 'Predicted', position: 'top' }} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <table className="price-table" style={{width: '100%', marginBottom: 16}}>
            <thead>
              <tr>
                <th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th><th>Sentiment</th><th>News</th>
              </tr>
            </thead>
            <tbody>
              {SAMPLE_PRICES.map((row, idx) => (
                <tr key={idx}>
                  <td>{row.date}</td>
                  <td>{row.open}</td>
                  <td>{row.high}</td>
                  <td>{row.low}</td>
                  <td>{row.close}</td>
                  <td>{row.volume}</td>
                  <td>{row.avg_sentiment_score}</td>
                  <td>{row.news_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="prediction-result">
            <strong>Predicted % Change (next day): </strong>
            {prediction !== null ? (prediction * 100).toFixed(2) + '%' : 'N/A'}
            {predictedClose !== null && (
              <span style={{ marginLeft: 16 }}>
                <strong>Predicted Close: </strong>{predictedClose.toFixed(2)}
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default StockChart; 