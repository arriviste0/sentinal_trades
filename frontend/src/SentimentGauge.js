import React, { useEffect, useState } from 'react';
import { getSentiment } from './api';

const SAMPLE_NEWS = [
  'Apple launches new AI-powered iPhone',
  'Tesla stock surges after record deliveries',
  'Federal Reserve signals possible rate cut',
];

function SentimentGauge() {
  const [avgSentiment, setAvgSentiment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchSentiments() {
      setLoading(true);
      setError('');
      try {
        const results = await Promise.all(SAMPLE_NEWS.map(headline => getSentiment(headline)));
        // Use positive score as a proxy for gauge (or you can use a custom formula)
        const avg = results.reduce((sum, r) => sum + (r.scores.positive - r.scores.negative), 0) / results.length;
        setAvgSentiment(avg);
      } catch (err) {
        setError('Error fetching sentiment.');
      } finally {
        setLoading(false);
      }
    }
    fetchSentiments();
  }, []);

  return (
    <div className="sentiment-gauge">
      <h2>Sentiment Gauge</h2>
      {loading && <div className="gauge-placeholder">Loading sentiment...</div>}
      {error && <div className="gauge-placeholder" style={{color: 'red'}}>{error}</div>}
      {!loading && !error && (
        <>
          <div className="gauge-value">
            <strong>Avg Sentiment (pos-neg): </strong>
            {avgSentiment !== null ? avgSentiment.toFixed(2) : 'N/A'}
          </div>
          <div className="gauge-placeholder">[Gauge chart will appear here]</div>
        </>
      )}
    </div>
  );
}

export default SentimentGauge; 