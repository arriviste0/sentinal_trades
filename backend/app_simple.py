from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
import random
import numpy as np
import pickle
import os
import json
import pandas as pd
from datetime import datetime
import openpyxl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai
from sklearn.metrics import accuracy_score
from collections import Counter
import re

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBE9-250SQ3mb8yBUVVa0svZSkU0HIrbho"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# Deep Learning Model for Sentiment Analysis
class DeepSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.max_len = 100
        self.vocab_size = 10000
        self.model_path = 'sentiment_model.h5'
        self.tokenizer_path = 'tokenizer.pkl'
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
                self.model = tf.keras.models.load_model(self.model_path)
                with open(self.tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print("Deep Learning Model loaded successfully")
            else:
                self.train_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model()
    
    def train_model(self):
        """Train the deep learning sentiment analysis model"""
        print("Training new Deep Learning model...")
        
        # Enhanced training data for financial sentiment analysis
        training_data = [
            # Positive samples (50)
            ("Reliance Industries reports record quarterly profits", "positive"),
            ("TCS wins $2 billion deal from European client", "positive"),
            ("HDFC Bank's net profit rises 18% YoY", "positive"),
            ("Infosys announces 10% salary hike for employees", "positive"),
            ("Maruti Suzuki sales jump 15% in festive season", "positive"),
            ("Bharti Airtel adds 4 million subscribers in Q3", "positive"),
            ("Sun Pharma receives FDA approval for new drug", "positive"),
            ("L&T bags $1.5 billion infrastructure project", "positive"),
            ("ITC's FMCG business grows 12% this quarter", "positive"),
            ("Adani Ports handles record cargo volume", "positive"),
            ("Wipro expands partnership with major cloud provider", "positive"),
            ("Axis Bank's asset quality improves significantly", "positive"),
            ("Tata Motors EV sales double in last quarter", "positive"),
            ("Nestle India launches innovative health products", "positive"),
            ("Dr Reddy's enters new therapeutic segment", "positive"),
            ("ONGC discovers new oil field in Krishna-Godavari basin", "positive"),
            ("UltraTech Cement capacity expansion ahead of schedule", "positive"),
            ("Asian Paints gains market share in decorative segment", "positive"),
            ("Bajaj Auto exports reach all-time high", "positive"),
            ("Tech Mahindra wins digital transformation deals", "positive"),
            ("ICICI Bank's digital transactions grow 25%", "positive"),
            ("HUL reports strong volume growth across categories", "positive"),
            ("JSW Steel commissions new production facility", "positive"),
            ("Cipla's respiratory portfolio performs exceptionally", "positive"),
            ("BPCL announces major refinery upgrade plan", "positive"),
            ("Hero MotoCorp launches new electric scooter", "positive"),
            ("Kotak Mahindra Bank's NIM improves to 4.5%", "positive"),
            ("Dabur's healthcare products see surge in demand", "positive"),
            ("GAIL expands natural gas pipeline network", "positive"),
            ("Persistent Systems reports strong deal pipeline", "positive"),
            ("SBI's retail loan book grows 18% YoY", "positive"),
            ("Britannia expands biscuit market dominance", "positive"),
            ("Vedanta's aluminum production hits record", "positive"),
            ("Lupin receives ANDA approval from USFDA", "positive"),
            ("Indian Oil announces dividend payout", "positive"),
            ("Mphasis acquires digital transformation firm", "positive"),
            ("Ashok Leyland's truck sales surge 22%", "positive"),
            ("Biocon's biosimilars gain traction in US market", "positive"),
            ("Power Grid completes transmission project early", "positive"),
            ("Godrej Consumer expands in rural markets", "positive"),
            ("Eicher Motors' premium bikes see strong demand", "positive"),
            ("Divis Labs' API business grows steadily", "positive"),
            ("Shree Cement's capacity utilization improves", "positive"),
            ("NTPC's renewable energy portfolio expands", "positive"),
            ("Colgate-Palmolive gains toothpaste market share", "positive"),
            ("Aurobindo Pharma's injectables facility gets EU approval", "positive"),
            ("Adani Green commissions new solar plant", "positive"),
            ("Grasim Industries' VSF demand recovers", "positive"),
            ("IndusInd Bank's deposit growth accelerates", "positive"),
            ("Punjab National Bank reduces NPAs significantly", "positive"),
            
            # Negative samples (40)
            ("Vodafone Idea reports massive quarterly loss", "negative"),
            ("Yes Bank shares plunge after RBI restrictions", "negative"),
            ("Tata Steel's European operations face headwinds", "negative"),
            ("DHFL defaults on debt payments", "negative"),
            ("IL&FS financial troubles deepen", "negative"),
            ("Jet Airways grounds all flights", "negative"),
            ("PC Jeweller faces liquidity crisis", "negative"),
            ("Reliance Communications files for bankruptcy", "negative"),
            ("Suzlon Energy's debt restructuring fails", "negative"),
            ("Essar Steel's insolvency case delayed", "negative"),
            ("Fortis Healthcare faces fraud allegations", "negative"),
            ("CG Power board discovers financial irregularities", "negative"),
            ("Cox & Kings defaults on payments", "negative"),
            ("HDIL promoters arrested in PMC Bank case", "negative"),
            ("Kwality Ltd's auditors raise red flags", "negative"),
            ("Reliance Capital faces rating downgrade", "negative"),
            ("Sintex Industries' lenders reject revival plan", "negative"),
            ("Alok Industries' resolution process stalls", "negative"),
            ("JP Associates fails to meet debt obligations", "negative"),
            ("Amtek Auto's insolvency case drags on", "negative"),
            ("Bhushan Steel's new management faces challenges", "negative"),
            ("Lanco Infratech's liquidation continues", "negative"),
            ("Reliance Power's earnings disappoint", "negative"),
            ("Unitech's homebuyers protest delays", "negative"),
            ("Punj Lloyd's liquidation proceedings begin", "negative"),
            ("Gitanjali Gems faces fraud investigation", "negative"),
            ("RICOH India's accounting issues surface", "negative"),
            ("Vakrangee accused of financial misreporting", "negative"),
            ("S Kumars Nationwide's revival fails", "negative"),
            ("Deccan Chronicle's financial troubles continue", "negative"),
            ("Kingfisher Airlines' Mallya faces extradition", "negative"),
            ("Videocon Industries' debt resolution stalls", "negative"),
            ("ABG Shipyard's lenders take haircut", "negative"),
            ("Monnet Ispat's operations remain shut", "negative"),
            ("Electrosteel Steels' new owner struggles", "negative"),
            ("Jaypee Infratech's homebuyers await resolution", "negative"),
            ("Religare Enterprises faces governance issues", "negative"),
            ("Eros International's financials questioned", "negative"),
            ("GVK Power's airport bid faces hurdles", "negative"),
            ("Dewan Housing's liquidity crisis worsens", "negative"),
            
            # Neutral samples (30)
            ("RBI keeps repo rate unchanged", "neutral"),
            ("SEBI introduces new disclosure norms", "neutral"),
            ("Government announces infrastructure spending", "neutral"),
            ("Monsoon rains arrive on schedule", "neutral"),
            ("Global oil prices remain stable", "neutral"),
            ("Rupee trades in narrow range", "neutral"),
            ("Nifty 50 index rebalancing announced", "neutral"),
            ("FPIs continue to invest in Indian markets", "neutral"),
            ("GST Council meets to review rates", "neutral"),
            ("India's GDP growth forecast maintained", "neutral"),
            ("Corporate tax collection meets targets", "neutral"),
            ("Trade deficit narrows slightly", "neutral"),
            ("Inflation remains within RBI's target", "neutral"),
            ("Bank credit growth shows modest increase", "neutral"),
            ("Railway budget focuses on modernization", "neutral"),
            ("Electric vehicle policy framework released", "neutral"),
            ("Renewable energy capacity addition on track", "neutral"),
            ("Manufacturing PMI shows steady growth", "neutral"),
            ("Services sector activity remains stable", "neutral"),
            ("Agricultural output estimates revised", "neutral"),
            ("Export growth slows marginally", "neutral"),
            ("Import duties on certain items reduced", "neutral"),
            ("FDI inflows maintain steady pace", "neutral"),
            ("Retail inflation eases slightly", "neutral"),
            ("Industrial production growth moderates", "neutral"),
            ("Monetary policy committee meeting concludes", "neutral"),
            ("Government announces divestment roadmap", "neutral"),
            ("Corporate bond issuances increase", "neutral"),
            ("Banking sector NPAs show marginal decline", "neutral"),
            ("Economic survey highlights growth challenges", "neutral")
        ]
        
        # Prepare training data
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # Convert labels to numerical values
        label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        numerical_labels = [label_map[label] for label in labels]
        
        # Tokenize text
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # Build LSTM model
        self.model = Sequential([
            Embedding(self.vocab_size, 64, input_length=self.max_len),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(24, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Train the model
        self.model.fit(
            padded_sequences,
            np.array(numerical_labels),
            epochs=50,
            verbose=1
        )
        
        # Save the model
        self.model.save(self.model_path)
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print("Deep Learning Model trained and saved successfully")
    
    def predict_sentiment(self, text):
        """Predict sentiment using deep learning model"""
        try:
            # Tokenize and pad text
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
            
            # Get prediction
            prediction = self.model.predict(padded)
            predicted_class = np.argmax(prediction, axis=1)[0]
            probabilities = prediction[0]
            
            # Convert numerical prediction back to label
            label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
            sentiment = label_map[predicted_class]
            
            # Calculate confidence
            confidence = max(probabilities) * 100
            
            return sentiment, confidence, probabilities
            
        except Exception as e:
            print(f"Deep Learning prediction error: {e}")
            return None, None, None

# Initialize the deep learning analyzer
deep_analyzer = DeepSentimentAnalyzer()

# Gemini LLM Sentiment Analyzer
class GeminiSentimentAnalyzer:
    def predict_sentiment(self, text):
        """Predict sentiment using Gemini LLM"""
        try:
            prompt = f"""
            Analyze the sentiment of the following financial news text and respond ONLY with one word: 
            'positive', 'negative', or 'neutral'. Do not include any other text or explanation.
            
            Text: "{text}"
            """
            
            response = gemini_model.generate_content(prompt)
            sentiment = response.text.strip().lower()
            
            # Validate response
            if sentiment not in ['positive', 'negative', 'neutral']:
                return None, None
            
            # For Gemini, we use a fixed confidence as it doesn't provide probabilities
            return sentiment, 85.0  # Fixed confidence for simplicity
            
        except Exception as e:
            print(f"Gemini prediction error: {e}")
            return None, None

# Initialize Gemini analyzer
gemini_analyzer = GeminiSentimentAnalyzer()

# Simple Company Extractor (without spaCy)
class SimpleCompanyExtractor:
    def __init__(self):
        # List of Indian companies
        self.common_companies = [
            "HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra Bank", 
            "TCS", "Infosys", "Wipro", "HCL Technologies", "Tech Mahindra",
            "Maruti Suzuki", "Tata Motors", "Mahindra & Mahindra", "Bajaj Auto",
            "Hindustan Unilever", "ITC", "Nestle India", "Britannia", 
            "Sun Pharma", "Dr Reddy's", "Cipla", "Lupin", "Biocon",
            "Reliance Industries", "ONGC", "Indian Oil", "BPCL", "HPCL",
            "Larsen & Toubro", "Adani Ports", "UltraTech Cement", "Ambuja Cements",
            "Bharti Airtel", "Vodafone Idea", "Reliance Jio",
            "Tata Steel", "JSW Steel", "Vedanta", "Hindalco", "Tata Power"
        ]
        
        # Sector keywords
        self.sector_keywords = {
            "Banking/Finance": ["bank", "finance", "loan", "credit", "lending", "nbfc"],
            "IT/Tech": ["software", "technology", "it services", "cloud", "digital", "ai"],
            "Automobile": ["auto", "vehicle", "car", "bike", "suv", "ev", "electric vehicle"],
            "FMCG": ["fmcg", "consumer", "goods", "retail", "packaged"],
            "Pharma/Healthcare": ["pharma", "medicine", "drug", "healthcare", "vaccine"],
            "Energy/Oil & Gas": ["oil", "gas", "energy", "petroleum", "renewable"],
            "Infrastructure": ["construction", "infra", "cement", "road", "highway"],
            "Telecom": ["telecom", "5g", "mobile", "broadband", "spectrum"]
        }
    
    def extract_companies(self, text):
        """Extract companies using simple keyword matching"""
        companies = []
        text_lower = text.lower()
        
        # Find companies in text
        for company in self.common_companies:
            if company.lower() in text_lower:
                companies.append(company)
        
        # Find sector
        sector_counts = {sector: 0 for sector in self.sector_keywords}
        for sector, keywords in self.sector_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    sector_counts[sector] += 1
        
        # Find primary sector
        primary_sector = max(sector_counts.items(), key=lambda x: x[1])[0]
        if max(sector_counts.values()) == 0:
            primary_sector = "General"
        
        return companies[:3] if companies else ["General Market"], primary_sector

# Initialize company extractor
company_extractor = SimpleCompanyExtractor()

# Data Storage and Model Management
class DataManager:
    def __init__(self):
        self.data_file = 'scraped_news_data.xlsx'
        self.training_data_file = 'training_data.xlsx'
        self.ensure_data_files()
    
    def ensure_data_files(self):
        """Create Excel files if they don't exist"""
        # Create scraped data file
        if not os.path.exists(self.data_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'title', 'description', 'url', 'image_url', 
                'companies', 'sector', 'deep_sentiment', 'deep_confidence', 'gemini_sentiment', 'gemini_confidence',
                'trend', 'price_prediction', 'ml_probabilities'
            ])
            df.to_excel(self.data_file, index=False)
            print(f"Created {self.data_file}")
        
        # Create training data file
        if not os.path.exists(self.training_data_file):
            df = pd.DataFrame(columns=[
                'text', 'sentiment', 'confidence', 'timestamp', 'source'
            ])
            df.to_excel(self.training_data_file, index=False)
            print(f"Created {self.training_data_file}")
    
    def store_scraped_data(self, news_data, analysis_results):
        """Store scraped news data with analysis results"""
        try:
            # Load existing data
            if os.path.exists(self.data_file):
                df = pd.read_excel(self.data_file)
            else:
                df = pd.DataFrame(columns=[
                    'timestamp', 'title', 'description', 'url', 'image_url', 
                    'companies', 'sector', 'deep_sentiment', 'deep_confidence', 'gemini_sentiment', 'gemini_confidence',
                    'trend', 'price_prediction', 'ml_probabilities'
                ])
            
            # Add new data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for i, article in enumerate(news_data):
                analysis = analysis_results.get(i, {})
                new_row = {
                    'timestamp': timestamp,
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'image_url': article.get('image_url', ''),
                    'companies': ', '.join(analysis.get('companies', [])),
                    'sector': analysis.get('sector', ''),
                    'deep_sentiment': analysis.get('deep_sentiment', ''),
                    'deep_confidence': analysis.get('deep_confidence', 0),
                    'gemini_sentiment': analysis.get('gemini_sentiment', ''),
                    'gemini_confidence': analysis.get('gemini_confidence', 0),
                    'trend': analysis.get('trend', ''),
                    'price_prediction': str(analysis.get('price_predictions', {})),
                    'ml_probabilities': str(analysis.get('deep_probabilities', []))
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save to Excel
            df.to_excel(self.data_file, index=False)
            print(f"Stored {len(news_data)} articles in {self.data_file}")
            
        except Exception as e:
            print(f"Error storing data: {e}")

# Initialize data manager
data_manager = DataManager()

def analyze_sentiment(text):
    """Advanced sentiment analysis using both models"""
    # Deep Learning analysis
    deep_sentiment, deep_confidence, deep_probabilities = deep_analyzer.predict_sentiment(text)
    
    # Gemini analysis
    gemini_sentiment, gemini_confidence = gemini_analyzer.predict_sentiment(text)
    
    # Enhanced keyword dictionaries for additional context
    positive_words = ['rise', 'gain', 'jump', 'surge', 'rally', 'up', 'positive', 'bullish', 'growth',
                     'profit', 'earnings', 'beat', 'exceed', 'strong', 'record', 'high', 'premium',
                     'overweight', 'buy', 'outperform', 'upgrade', 'positive', 'favorable', 'success']
    
    negative_words = ['fall', 'drop', 'decline', 'crash', 'down', 'negative', 'bearish', 'loss',
                     'miss', 'below', 'weak', 'concern', 'risk', 'short', 'sell', 'underweight',
                     'downgrade', 'negative', 'unfavorable', 'ponzi', 'allegations', 'failure']
    
    # Calculate keyword scores
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    return {
        'deep_sentiment': deep_sentiment,
        'deep_confidence': deep_confidence,
        'deep_probabilities': deep_probabilities,
        'gemini_sentiment': gemini_sentiment,
        'gemini_confidence': gemini_confidence,
        'positive_keywords': positive_count,
        'negative_keywords': negative_count
    }

def predict_trend(sentiment, confidence, positive_count, negative_count):
    """Advanced trend prediction with timeline and impact analysis"""
    sentiment_score = 0
    if sentiment == 'positive':
        sentiment_score = confidence * (1 + positive_count / 10)
    elif sentiment == 'negative':
        sentiment_score = -confidence * (1 + negative_count / 10)
    
    if sentiment_score > 80:
        return 'upward', 'strong', '1-3 days', 'High impact expected'
    elif sentiment_score > 60:
        return 'upward', 'moderate', '3-7 days', 'Moderate impact expected'
    elif sentiment_score > 40:
        return 'upward', 'weak', '1-2 weeks', 'Gradual positive movement'
    elif sentiment_score < -80:
        return 'downward', 'strong', '1-3 days', 'High impact expected'
    elif sentiment_score < -60:
        return 'downward', 'moderate', '3-7 days', 'Moderate impact expected'
    elif sentiment_score < -40:
        return 'downward', 'weak', '1-2 weeks', 'Gradual negative movement'
    else:
        return 'neutral', 'minimal', '1-4 weeks', 'Limited immediate impact'

def predict_price_change(sentiment, confidence, trend_strength):
    """Advanced price prediction with multiple scenarios"""
    if sentiment == 'positive':
        if trend_strength == 'strong':
            base_change = 2.5 + (confidence - 70) * 0.08
            return f"+{min(8.0, base_change):.1f}%", f"+{min(12.0, base_change * 1.5):.1f}%", f"+{min(15.0, base_change * 2):.1f}%"
        elif trend_strength == 'moderate':
            base_change = 1.5 + (confidence - 70) * 0.05
            return f"+{min(5.0, base_change):.1f}%", f"+{min(7.0, base_change * 1.3):.1f}%", f"+{min(10.0, base_change * 1.8):.1f}%"
        else:
            base_change = 0.8 + (confidence - 70) * 0.03
            return f"+{min(3.0, base_change):.1f}%", f"+{min(4.5, base_change * 1.2):.1f}%", f"+{min(6.0, base_change * 1.5):.1f}%"
    elif sentiment == 'negative':
        if trend_strength == 'strong':
            base_change = 2.0 + (confidence - 70) * 0.06
            return f"-{min(6.0, base_change):.1f}%", f"-{min(9.0, base_change * 1.4):.1f}%", f"-{min(12.0, base_change * 1.8):.1f}%"
        elif trend_strength == 'moderate':
            base_change = 1.2 + (confidence - 70) * 0.04
            return f"-{min(4.0, base_change):.1f}%", f"-{min(6.0, base_change * 1.3):.1f}%", f"-{min(8.0, base_change * 1.6):.1f}%"
        else:
            base_change = 0.6 + (confidence - 70) * 0.02
            return f"-{min(2.5, base_change):.1f}%", f"-{min(3.5, base_change * 1.2):.1f}%", f"-{min(5.0, base_change * 1.4):.1f}%"
    else:
        return "±0.5%", "±1.0%", "±1.5%"

@app.route('/analyze-news', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        title = data.get('title', '')
        description = data.get('description', '')
        
        # Combine text for analysis
        full_text = f"{title} {description}"
        
        # Extract companies and sector
        companies, sector = company_extractor.extract_companies(full_text)
        
        # Perform sentiment analysis with both models
        sentiment_results = analyze_sentiment(full_text)
        
        # Extract results
        deep_sentiment = sentiment_results['deep_sentiment']
        deep_confidence = sentiment_results['deep_confidence']
        gemini_sentiment = sentiment_results['gemini_sentiment']
        gemini_confidence = sentiment_results['gemini_confidence']
        
        # Predict trend with timeline (using deep learning model as primary)
        trend, trend_strength, timeline, impact_description = predict_trend(
            deep_sentiment, 
            deep_confidence, 
            sentiment_results['positive_keywords'],
            sentiment_results['negative_keywords']
        )
        
        # Predict price changes for different timeframes
        short_term, medium_term, long_term = predict_price_change(
            deep_sentiment, 
            deep_confidence, 
            trend_strength
        )
        
        # Separate detailed analysis for each model
        deep_analysis = f"🔍 **Deep Learning Analysis:**\n"
        deep_analysis += f"• Sentiment: {deep_sentiment.upper()}\n"
        deep_analysis += f"• Confidence: {deep_confidence:.1f}%\n"
        deep_analysis += f"• Model: LSTM Neural Network\n"
        deep_analysis += f"• Architecture: Embedding → LSTM(64) → LSTM(32) → Dense(24) → Output(3)\n"
        deep_analysis += f"• Training: 120+ Indian financial news samples\n"
        deep_analysis += f"• Probabilities: Positive({sentiment_results['deep_probabilities'][0]*100:.1f}%), Negative({sentiment_results['deep_probabilities'][1]*100:.1f}%), Neutral({sentiment_results['deep_probabilities'][2]*100:.1f}%)\n"
        
        gemini_analysis = f"🤖 **Gemini LLM Analysis:**\n"
        gemini_analysis += f"• Sentiment: {gemini_sentiment.upper()}\n"
        gemini_analysis += f"• Confidence: {gemini_confidence:.1f}%\n"
        gemini_analysis += f"• Model: Gemini Pro\n"
        gemini_analysis += f"• Capabilities: Advanced language understanding\n"
        gemini_analysis += f"• Context: Financial news analysis\n"
        gemini_analysis += f"• Strengths: Nuance detection, contextual understanding\n"
        
        # Model comparison
        comparison = f"📊 **Model Comparison:**\n"
        if deep_sentiment == gemini_sentiment:
            comparison += f"✅ **AGREEMENT**: Both models predict {deep_sentiment.upper()} sentiment\n"
            comparison += f"• Agreement Level: HIGH\n"
            comparison += f"• Reliability: EXCELLENT\n"
            comparison += f"• Recommendation: Strong confidence in prediction\n"
        else:
            comparison += f"⚠️ **DISAGREEMENT**: Models have different predictions\n"
            comparison += f"• Deep Learning: {deep_sentiment.upper()} ({deep_confidence:.1f}% confidence)\n"
            comparison += f"• Gemini LLM: {gemini_sentiment.upper()} ({gemini_confidence:.1f}% confidence)\n"
            comparison += f"• Agreement Level: LOW\n"
            comparison += f"• Recommendation: Exercise caution, consider both perspectives\n"
        
        # Confidence comparison
        confidence_diff = abs(deep_confidence - gemini_confidence)
        if confidence_diff < 10:
            comparison += f"• Confidence Difference: {confidence_diff:.1f}% (Similar confidence levels)\n"
        else:
            comparison += f"• Confidence Difference: {confidence_diff:.1f}% (Significant difference)\n"
        
        # Overall analysis
        overall_analysis = f"📈 **Overall Market Analysis:**\n"
        overall_analysis += f"• Sector: {sector}\n"
        overall_analysis += f"• Companies: {', '.join(companies)}\n"
        overall_analysis += f"• Expected Trend: {trend} ({trend_strength} strength)\n"
        overall_analysis += f"• Timeline: {timeline}\n"
        overall_analysis += f"• Impact: {impact_description}\n"
        overall_analysis += f"• Price Predictions: Short-term ({short_term}), Medium-term ({medium_term}), Long-term ({long_term})\n"
        
        # Final recommendation
        if deep_sentiment == gemini_sentiment:
            if deep_sentiment == 'positive':
                overall_analysis += f"• Recommendation: Strong buy signal - both models agree on positive outlook\n"
            elif deep_sentiment == 'negative':
                overall_analysis += f"• Recommendation: Exercise caution - both models indicate potential risks\n"
            else:
                overall_analysis += f"• Recommendation: Neutral stance - limited immediate impact expected\n"
        else:
            overall_analysis += f"• Recommendation: Mixed signals - consider both perspectives before making decisions\n"

        # Prepare response with separate model analyses
        response_data = {
            'companies': companies,
            'sector': sector,
            'deep_sentiment': deep_sentiment,
            'deep_confidence': deep_confidence,
            'deep_probabilities': sentiment_results['deep_probabilities'].tolist(),
            'gemini_sentiment': gemini_sentiment,
            'gemini_confidence': gemini_confidence,
            'trend': trend,
            'trend_strength': trend_strength,
            'timeline': timeline,
            'impact_description': impact_description,
            'price_predictions': {
                'short_term': short_term,
                'medium_term': medium_term,
                'long_term': long_term
            },
            'deep_learning_analysis': deep_analysis,
            'gemini_llm_analysis': gemini_analysis,
            'model_comparison': comparison,
            'overall_analysis': overall_analysis,
            'disclaimer': 'Analysis for informational purposes only. Not financial advice.'
        }
        
        # Store results
        data_manager.store_scraped_data(
            [{'title': title, 'description': description}], 
            {0: response_data}
        )
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Stock News Analyzer (Simple Version)")
    print("✅ No spaCy required - using simple keyword matching")
    app.run(host='0.0.0.0', port=5000, debug=True) 