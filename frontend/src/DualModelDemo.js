import React from 'react';

const DualModelDemo = () => {
  const demoData = {
    companies: ["TCS", "Infosys"],
    sector: "IT/Tech",
    deep_sentiment: "positive",
    deep_confidence: 87.5,
    deep_probabilities: [0.75, 0.15, 0.10],
    gemini_sentiment: "positive",
    gemini_confidence: 85.0,
    trend: "upward",
    trend_strength: "moderate",
    timeline: "3-7 days",
    impact_description: "Moderate impact expected",
    price_predictions: {
      short_term: "+2.1%",
      medium_term: "+3.2%",
      long_term: "+4.8%"
    },
    deep_learning_analysis: `🔍 **Deep Learning Analysis:**
• Sentiment: POSITIVE
• Confidence: 87.5%
• Model: LSTM Neural Network
• Architecture: Embedding → LSTM(64) → LSTM(32) → Dense(24) → Output(3)
• Training: 120+ Indian financial news samples
• Probabilities: Positive(75.0%), Negative(15.0%), Neutral(10.0%)`,
    gemini_llm_analysis: `🤖 **Gemini LLM Analysis:**
• Sentiment: POSITIVE
• Confidence: 85.0%
• Model: Gemini Pro
• Capabilities: Advanced language understanding
• Context: Financial news analysis
• Strengths: Nuance detection, contextual understanding`,
    model_comparison: `📊 **Model Comparison:**
✅ **AGREEMENT**: Both models predict POSITIVE sentiment
• Agreement Level: HIGH
• Reliability: EXCELLENT
• Recommendation: Strong confidence in prediction
• Confidence Difference: 2.5% (Similar confidence levels)`,
    overall_analysis: `📈 **Overall Market Analysis:**
• Sector: IT/Tech
• Companies: TCS, Infosys
• Expected Trend: upward (moderate strength)
• Timeline: 3-7 days
• Impact: Moderate impact expected
• Price Predictions: Short-term (+2.1%), Medium-term (+3.2%), Long-term (+4.8%)
• Recommendation: Strong buy signal - both models agree on positive outlook`
  };

  return (
    <div className="analysis-results">
      <div className="analysis-header">
        <span className="analysis-icon">🔍</span>
        <h4>Dual AI Model Analysis</h4>
      </div>
      
      {/* Company and Sector Info */}
      <div className="company-sector-info">
        <div className="info-item">
          <span className="info-label">🏢 Companies:</span>
          <div className="company-tags">
            {demoData.companies.map((company, idx) => (
              <span key={idx} className="company-tag">{company}</span>
            ))}
          </div>
        </div>
        <div className="info-item">
          <span className="info-label">🏭 Sector:</span>
          <span className="sector-value">{demoData.sector}</span>
        </div>
      </div>

      {/* Dual Model Comparison */}
      <div className="dual-model-comparison">
        <div className="model-column deep-learning">
          <div className="model-header">
            <span className="model-icon">🔍</span>
            <span className="model-name">Deep Learning (LSTM)</span>
          </div>
          <div className="model-results">
            <div className="sentiment-result">
              <span className="result-label">Sentiment:</span>
              <span className={`sentiment-badge ${demoData.deep_sentiment}`}>
                {demoData.deep_sentiment}
              </span>
            </div>
            <div className="confidence-result">
              <span className="result-label">Confidence:</span>
              <span className="confidence-value">
                {demoData.deep_confidence}%
              </span>
            </div>
            <div className="probabilities">
              <span className="result-label">Probabilities:</span>
              <div className="prob-grid">
                <div className="prob-item">
                  <span className="prob-label">Positive:</span>
                  <span className="prob-value">{(demoData.deep_probabilities[0] * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-item">
                  <span className="prob-label">Negative:</span>
                  <span className="prob-value">{(demoData.deep_probabilities[1] * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-item">
                  <span className="prob-label">Neutral:</span>
                  <span className="prob-value">{(demoData.deep_probabilities[2] * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="model-column gemini">
          <div className="model-header">
            <span className="model-icon">🤖</span>
            <span className="model-name">Gemini LLM</span>
          </div>
          <div className="model-results">
            <div className="sentiment-result">
              <span className="result-label">Sentiment:</span>
              <span className={`sentiment-badge ${demoData.gemini_sentiment}`}>
                {demoData.gemini_sentiment}
              </span>
            </div>
            <div className="confidence-result">
              <span className="result-label">Confidence:</span>
              <span className="confidence-value">
                {demoData.gemini_confidence}%
              </span>
            </div>
            <div className="model-capabilities">
              <span className="result-label">Capabilities:</span>
              <div className="capabilities-list">
                <span className="capability">Advanced Language Understanding</span>
                <span className="capability">Contextual Analysis</span>
                <span className="capability">Nuance Detection</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Agreement Status */}
      <div className="agreement-status">
        <div className="agreement-header">
          <span className="agreement-icon">✅</span>
          <span className="agreement-title">Models Agree</span>
        </div>
        <div className="agreement-details">
          <div className="comparison-text">
            {demoData.model_comparison.split('\n').map((line, idx) => (
              <div key={idx} className="comparison-line">{line}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Market Analysis */}
      <div className="market-analysis">
        <div className="analysis-header">
          <span className="analysis-icon">📈</span>
          <h5>Market Analysis</h5>
        </div>
        <div className="analysis-grid">
          <div className="analysis-item">
            <span className="analysis-label">Trend:</span>
            <span className={`trend-badge ${demoData.trend}`}>
              {demoData.trend} ({demoData.trend_strength})
            </span>
          </div>
          <div className="analysis-item">
            <span className="analysis-label">Timeline:</span>
            <span className="timeline-value">{demoData.timeline}</span>
          </div>
          <div className="analysis-item">
            <span className="analysis-label">Impact:</span>
            <span className="impact-value">{demoData.impact_description}</span>
          </div>
        </div>
      </div>

      {/* Price Predictions */}
      <div className="price-predictions-section">
        <h5 className="section-title">Price Predictions</h5>
        <div className="price-grid">
          <div className="price-item">
            <span className="price-label">Short-term:</span>
            <span className={`price-value ${demoData.deep_sentiment}`}>
              {demoData.price_predictions.short_term}
            </span>
          </div>
          <div className="price-item">
            <span className="price-label">Medium-term:</span>
            <span className={`price-value ${demoData.deep_sentiment}`}>
              {demoData.price_predictions.medium_term}
            </span>
          </div>
          <div className="price-item">
            <span className="price-label">Long-term:</span>
            <span className={`price-value ${demoData.deep_sentiment}`}>
              {demoData.price_predictions.long_term}
            </span>
          </div>
        </div>
      </div>

      {/* Overall Analysis */}
      <div className="overall-analysis">
        <h5 className="section-title">Overall Analysis</h5>
        <div className="analysis-text">
          {demoData.overall_analysis.split('\n').map((line, idx) => (
            <div key={idx} className="analysis-line">{line}</div>
          ))}
        </div>
      </div>

      {/* Detailed Model Analysis */}
      <div className="detailed-analysis">
        <div className="analysis-tabs">
          <div className="tab-content">
            <div className="model-analysis">
              <h6>🔍 Deep Learning Analysis</h6>
              <div className="analysis-text">
                {demoData.deep_learning_analysis.split('\n').map((line, idx) => (
                  <div key={idx} className="analysis-line">{line}</div>
                ))}
              </div>
            </div>
            <div className="model-analysis">
              <h6>🤖 Gemini LLM Analysis</h6>
              <div className="analysis-text">
                {demoData.gemini_llm_analysis.split('\n').map((line, idx) => (
                  <div key={idx} className="analysis-line">{line}</div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="disclaimer-section">
        <div className="disclaimer-header">
          <span className="disclaimer-icon">⚠️</span>
          <span className="disclaimer-title">Important Disclaimer</span>
        </div>
        <p className="disclaimer-text">
          This analysis is for informational purposes only and should not be considered as financial advice. Always conduct your own research and consult with financial advisors before making investment decisions.
        </p>
      </div>
    </div>
  );
};

export default DualModelDemo; 