#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced dual-model analysis
"""

import requests
import json

def test_analysis():
    """Test the enhanced analysis with sample news"""
    
    # Sample news data
    test_cases = [
        {
            "title": "TCS reports strong quarterly results with 15% revenue growth",
            "description": "TCS beats analyst estimates with record profits and announces dividend payout. The company's digital transformation business shows exceptional growth."
        },
        {
            "title": "Vodafone Idea faces financial crisis and defaults on payments",
            "description": "Vodafone Idea reports massive quarterly loss and struggles with debt obligations. The telecom sector faces regulatory challenges."
        },
        {
            "title": "RBI keeps repo rate unchanged at 6.5%",
            "description": "The Reserve Bank of India maintains current interest rates citing stable inflation. Monetary policy committee decision aligns with market expectations."
        }
    ]
    
    print("🚀 Testing Enhanced Dual-Model Analysis")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📰 Test Case {i}: {test_case['title']}")
        print("-" * 50)
        
        try:
            # Send request to the API
            response = requests.post(
                'http://localhost:5000/analyze-news',
                json=test_case,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display results
                print(f"🏢 Companies: {', '.join(data['companies'])}")
                print(f"🏭 Sector: {data['sector']}")
                print()
                
                # Deep Learning Analysis
                print(data['deep_learning_analysis'])
                
                # Gemini LLM Analysis
                print(data['gemini_llm_analysis'])
                
                # Model Comparison
                print(data['model_comparison'])
                
                # Overall Analysis
                print(data['overall_analysis'])
                
                print(f"📊 Deep Learning Probabilities: {data['deep_probabilities']}")
                print(f"💰 Price Predictions: {data['price_predictions']}")
                
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the server is running (python app.py)")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    test_analysis() 