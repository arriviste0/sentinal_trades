import React, { useEffect, useState } from 'react';
import { getSentiment, getContext, getEntities } from './api';

// Sample static news for demo; replace with real API/news source later
const SAMPLE_NEWS = [
  {
    id: 1,
    headline: 'Apple launches new AI-powered iPhone',
    timestamp: '2024-07-10T09:00:00Z',
  },
  {
    id: 2,
    headline: 'Tesla stock surges after record deliveries',
    timestamp: '2024-07-10T10:30:00Z',
  },
  {
    id: 3,
    headline: 'Federal Reserve signals possible rate cut',
    timestamp: '2024-07-10T11:15:00Z',
  },
];

function NewsFeed() {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function fetchAndAnalyze() {
    setLoading(true);
    setError('');
    try {
      // For each news item, fetch sentiment, context, and entities
      const analyzed = await Promise.all(
        SAMPLE_NEWS.map(async (item) => {
          const [sentiment, context, entities] = await Promise.all([
            getSentiment(item.headline),
            getContext(item.headline),
            getEntities(item.headline),
          ]);
          return {
            ...item,
            sentiment,
            context,
            entities: entities.entities || [],
          };
        })
      );
      setNews(analyzed);
    } catch (err) {
      setError('Error fetching news analysis.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchAndAnalyze();
    // eslint-disable-next-line
  }, []);

  return (
    <div className="news-feed">
      <h2>Live News Feed</h2>
      <button onClick={fetchAndAnalyze} disabled={loading} style={{marginBottom: 16}}>
        {loading ? 'Fetching & Analyzing...' : 'Fetch Latest News'}
      </button>
      {loading && <div className="news-placeholder">Loading news...</div>}
      {error && <div className="news-placeholder" style={{color: 'red'}}>{error}</div>}
      {!loading && !error && news.length === 0 && (
        <div className="news-placeholder">No news available.</div>
      )}
      {!loading && !error && news.length > 0 && (
        <ul className="news-list">
          {news.map((item) => (
            <li key={item.id} className="news-item">
              <div className="news-headline">{item.headline}</div>
              <div className="news-meta">
                <span className={`sentiment-label sentiment-${item.sentiment.label}`}>{item.sentiment.label}</span>
                <span className="sector-label">Sector: {item.context.sector}</span>
                <span className="impact-label">Impact: {item.context.impact}</span>
              </div>
              <div className="entities-list">
                Entities: {item.entities.map(e => `${e.text} (${e.label})`).join(', ') || 'None'}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default NewsFeed; 