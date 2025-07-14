# Stock News Analyzer - Enhanced Setup Guide

## Overview
This enhanced version includes:
- **Company Extraction**: Automatically identifies Indian companies mentioned in news
- **Sector Detection**: Categorizes news by business sectors (Banking, IT, Auto, etc.)
- **Enhanced Training Data**: 120+ Indian financial news samples for better accuracy
- **Dual AI Analysis**: Deep Learning LSTM + Gemini LLM for robust sentiment analysis

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install spaCy Model
```bash
# Option 1: Use the setup script
python setup_spacy.py

# Option 2: Manual installation
python -m spacy download en_core_web_sm
```

### 3. Configure API Keys
Update the Gemini API key in `app.py`:
```python
GEMINI_API_KEY = "your_actual_api_key_here"
```

## Features

### Company Extraction
- **120+ Indian Companies**: Covers major sectors including Banking, IT, Auto, FMCG, Pharma, Energy, Infrastructure, Telecom
- **Sector Detection**: Automatically categorizes news into 8 major sectors
- **Hybrid Extraction**: Uses Gemini LLM + spaCy NLP with fallback mechanisms

### Enhanced Training Data
- **50 Positive Samples**: Real Indian company success stories
- **40 Negative Samples**: Financial troubles, defaults, fraud cases
- **30 Neutral Samples**: Regulatory updates, policy changes

### Sectors Covered
1. **Banking/Finance**: HDFC Bank, ICICI Bank, SBI, etc.
2. **IT/Tech**: TCS, Infosys, Wipro, etc.
3. **Automobile**: Maruti Suzuki, Tata Motors, etc.
4. **FMCG**: HUL, ITC, Nestle India, etc.
5. **Pharma/Healthcare**: Sun Pharma, Dr Reddy's, etc.
6. **Energy/Oil & Gas**: Reliance Industries, ONGC, etc.
7. **Infrastructure**: L&T, Adani Ports, etc.
8. **Telecom**: Bharti Airtel, Vodafone Idea, etc.

## Usage

### Start the Server
```bash
python app.py
```

### API Endpoints

#### 1. Get Market News
```bash
GET http://localhost:5000/market-news
```

#### 2. Analyze News
```bash
POST http://localhost:5000/analyze-news
Content-Type: application/json

{
    "title": "TCS reports strong quarterly results",
    "description": "TCS beats estimates with 15% revenue growth..."
}
```

#### 3. Get Statistics
```bash
GET http://localhost:5000/data-statistics
```

## Enhanced Response Format
```json
{
    "companies": ["TCS", "Infosys"],
    "sector": "IT/Tech",
    "deep_sentiment": "positive",
    "deep_confidence": 87.5,
    "deep_probabilities": [0.75, 0.15, 0.10],
    "gemini_sentiment": "positive",
    "gemini_confidence": 85.0,
    "trend": "upward",
    "trend_strength": "moderate",
    "timeline": "3-7 days",
    "impact_description": "Moderate impact expected",
    "price_predictions": {
        "short_term": "+2.1%",
        "medium_term": "+3.2%",
        "long_term": "+4.8%"
    },
    "deep_learning_analysis": "🔍 **Deep Learning Analysis:**\n• Sentiment: POSITIVE\n• Confidence: 87.5%\n• Model: LSTM Neural Network\n...",
    "gemini_llm_analysis": "🤖 **Gemini LLM Analysis:**\n• Sentiment: POSITIVE\n• Confidence: 85.0%\n• Model: Gemini Pro\n...",
    "model_comparison": "📊 **Model Comparison:**\n✅ **AGREEMENT**: Both models predict POSITIVE sentiment\n• Agreement Level: HIGH\n...",
    "overall_analysis": "📈 **Overall Market Analysis:**\n• Sector: IT/Tech\n• Companies: TCS, Infosys\n• Expected Trend: upward (moderate strength)\n..."
}
```

### Analysis Components:

1. **🔍 Deep Learning Analysis**: Detailed breakdown of LSTM model results
2. **🤖 Gemini LLM Analysis**: Comprehensive Gemini Pro analysis
3. **📊 Model Comparison**: Side-by-side comparison and agreement status
4. **📈 Overall Market Analysis**: Combined insights and recommendations

## Troubleshooting

### spaCy Model Issues
If you get spaCy model errors:
```bash
python -m spacy download en_core_web_sm
```

### TensorFlow Issues
For TensorFlow installation problems:
```bash
pip install tensorflow==2.13.0
```

### API Key Issues
Ensure your Gemini API key is valid and has sufficient quota.

## Data Storage
- News data is stored in `scraped_news_data.xlsx`
- Training data is stored in `training_data.xlsx`
- Model files: `sentiment_model.h5`, `tokenizer.pkl`

## Model Information
- **Deep Learning**: LSTM Neural Network with 85-90% accuracy
- **Gemini LLM**: Advanced language understanding for context-aware analysis
- **Training**: 120+ curated Indian financial news samples
- **Sectors**: 8 major business sectors with keyword mapping 