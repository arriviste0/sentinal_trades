import React, { useEffect, useState } from 'react';

// Sample price and news data (reuse from StockChart)
const SAMPLE_PRICES = [
  { date: '2024-07-05', close: 152, news_count: 3 },
  { date: '2024-07-06', close: 153, news_count: 2 },
  { date: '2024-07-07', close: 154, news_count: 4 },
  { date: '2024-07-08', close: 155, news_count: 5 },
  { date: '2024-07-09', close: 156, news_count: 6 },
];

function RiskMeter() {
  const [risk, setRisk] = useState(null);

  useEffect(() => {
    // Calculate volatility (std dev of close price)
    const closes = SAMPLE_PRICES.map(d => d.close);
    const mean = closes.reduce((a, b) => a + b, 0) / closes.length;
    const variance = closes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / closes.length;
    const volatility = Math.sqrt(variance);
    // News volume (average news per day)
    const avgNews = SAMPLE_PRICES.reduce((a, b) => a + b.news_count, 0) / SAMPLE_PRICES.length;
    // Simple risk score: weighted sum (scale as needed)
    const riskScore = (volatility * 10) + (avgNews * 2);
    setRisk(riskScore);
  }, []);

  return (
    <div className="risk-meter">
      <h2>Risk Meter</h2>
      <div className="risk-value">
        <strong>Risk Score: </strong>
        {risk !== null ? risk.toFixed(2) : 'N/A'}
      </div>
      <div className="risk-placeholder">[Risk meter will appear here]</div>
    </div>
  );
}

export default RiskMeter; 