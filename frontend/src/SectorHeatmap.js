import React, { useEffect, useState } from 'react';
import { getContext, getSentiment } from './api';

const SAMPLE_NEWS = [
  'Apple launches new AI-powered iPhone',
  'Tesla stock surges after record deliveries',
  'Federal Reserve signals possible rate cut',
  'Pfizer announces breakthrough in cancer drug',
  'Chevron expands renewable energy investments',
];

function SectorHeatmap() {
  const [sectorSentiment, setSectorSentiment] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchSectorSentiment() {
      setLoading(true);
      setError('');
      try {
        // For each news, get sector and sentiment
        const results = await Promise.all(
          SAMPLE_NEWS.map(async (headline) => {
            const [context, sentiment] = await Promise.all([
              getContext(headline),
              getSentiment(headline),
            ]);
            return {
              sector: context.sector || 'Unknown',
              sentiment: sentiment.scores.positive - sentiment.scores.negative,
            };
          })
        );
        // Aggregate by sector
        const sectorMap = {};
        results.forEach(({ sector, sentiment }) => {
          if (!sectorMap[sector]) sectorMap[sector] = [];
          sectorMap[sector].push(sentiment);
        });
        const agg = Object.entries(sectorMap).map(([sector, sentiments]) => ({
          sector,
          avgSentiment: sentiments.reduce((a, b) => a + b, 0) / sentiments.length,
        }));
        setSectorSentiment(agg);
      } catch (err) {
        setError('Error fetching sector sentiment.');
      } finally {
        setLoading(false);
      }
    }
    fetchSectorSentiment();
  }, []);

  return (
    <div className="sector-heatmap">
      <h2>Sector-wise Sentiment Heatmap</h2>
      {loading && <div className="heatmap-placeholder">Loading sector sentiment...</div>}
      {error && <div className="heatmap-placeholder" style={{color: 'red'}}>{error}</div>}
      {!loading && !error && sectorSentiment.length === 0 && (
        <div className="heatmap-placeholder">No sector data.</div>
      )}
      {!loading && !error && sectorSentiment.length > 0 && (
        <table className="sector-table" style={{width: '100%'}}>
          <thead>
            <tr><th>Sector</th><th>Avg Sentiment</th></tr>
          </thead>
          <tbody>
            {sectorSentiment.map((row, idx) => (
              <tr key={idx}>
                <td>{row.sector}</td>
                <td>{row.avgSentiment.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="heatmap-placeholder">[Heatmap will appear here]</div>
    </div>
  );
}

export default SectorHeatmap; 