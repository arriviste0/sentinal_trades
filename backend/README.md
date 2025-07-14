# Stock News Hub 📈

A modern, responsive web application that displays the latest stock market news with a beautiful UI. The app scrapes financial news from MoneyControl and presents them in an elegant card-based layout.

## Features

- 🎨 **Modern UI Design**: Beautiful gradient background with glassmorphism effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- 🔍 **Search Functionality**: Filter news articles by title or description
- 📊 **Real-time Stats**: Display article count and filtering statistics
- 🏢 **Company Detection**: Automatically identify Indian stocks mentioned in news
- 📈 **AI Analysis**: Perform sentiment analysis and price predictions
- 🎯 **Trend Prediction**: Predict upward/downward/neutral market trends
- 🖼️ **Image Support**: News articles with featured images
- ⚡ **Fast Loading**: Optimized performance with smooth animations
- 🔗 **Direct Links**: Click to read full articles on the source website

## Tech Stack

### Backend
- **Flask**: Python web framework
- **BeautifulSoup**: Web scraping library
- **Flask-CORS**: Cross-origin resource sharing
- **Requests**: HTTP library for making requests
- **Scikit-learn**: Machine learning library for sentiment analysis
- **Random Forest**: ML model for sentiment classification
- **TF-IDF**: Text feature extraction for ML model
- **Pandas**: Data manipulation and Excel file handling
- **OpenPyXL**: Excel file operations for data storage

### Frontend
- **React**: JavaScript library for building user interfaces
- **CSS3**: Modern styling with gradients, animations, and responsive design
- **HTML5**: Semantic markup

## Installation & Setup

### Prerequisites
- Python 3.7+
- Node.js 14+
- npm or yarn

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask server**:
   ```bash
   python app.py
   ```
   
   The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```
   
   The frontend will start on `http://localhost:3000`

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Click the "Fetch Latest News" button to load the latest stock news
3. Use the search bar to filter articles by keywords
4. Click on any article title or "Read Full Article" button to open the original source
5. View real-time statistics about the number of articles and filtering status

## API Endpoints

### GET `/market-news`
Returns the latest stock market news articles.

**Response Format**:
```json
{
  "status": "success",
  "count": 25,
  "news": [
    {
      "title": "Article Title",
      "description": "Article description...",
      "image_url": "https://example.com/image.jpg",
      "url": "https://example.com/article"
    }
  ]
}
```

### POST `/analyze-news`
Performs sentiment analysis and price prediction for news articles.

**Request Body**:
```json
{
  "title": "Article title",
  "description": "Article description",
  "companies": ["TCS", "Infosys"]
}
```

**Response Format**:
```json
{
  "companies": ["TCS", "Infosys"],
  "sentiment": "positive",
  "trend": "upward",
  "confidence": 85,
  "price_prediction": "+2.5%",
  "analysis": "Based on the news content, TCS shows a positive sentiment with 85% confidence..."
}
```

### GET `/data-statistics`
Returns statistics about stored data and model performance.

**Response Format**:
```json
{
  "status": "success",
  "statistics": {
    "total_articles": 150,
    "unique_companies": 25,
    "sentiment_distribution": {"positive": 60, "negative": 30, "neutral": 60},
    "avg_confidence": 82.5,
    "training_samples": 45,
    "last_updated": "2024-01-15 14:30:00"
  }
}
```

### POST `/retrain-model`
Manually triggers model retraining with accumulated data.

**Response Format**:
```json
{
  "status": "success",
  "message": "Model retraining completed"
}
```

### POST `/add-training-data`
Adds manual training data for model improvement.

**Request Body**:
```json
{
  "text": "Stock price rises sharply after strong earnings",
  "sentiment": "positive",
  "confidence": 85,
  "source": "manual"
}
```

## Project Structure

```
stock_news/
├── app.py                 # Flask backend server
├── frontend/             # React frontend application
│   ├── public/           # Static files
│   ├── src/              # React source code
│   │   ├── App.js        # Main React component
│   │   ├── App.css       # Main styles
│   │   ├── index.js      # React entry point
│   │   └── index.css     # Global styles
│   ├── package.json      # Frontend dependencies
│   └── README.md         # Frontend documentation
└── README.md             # Project documentation
```

## Features in Detail

### 🎨 Modern Design
- Gradient background with purple-blue theme
- Glassmorphism effects with backdrop blur
- Smooth hover animations and transitions
- Professional typography and spacing

### 📱 Responsive Layout
- Mobile-first design approach
- Adaptive grid layout for news cards
- Flexible header and navigation
- Optimized for all screen sizes

### 🔍 Search & Filter
- Real-time search functionality
- Filters by article title and description
- Dynamic result count updates
- Clear search interface with icons

### 🏢 Company Detection & Analysis
- Automatic identification of Indian stocks in news content
- Comprehensive database of Indian companies and stocks
- Real-time company tagging for each news article
- Smart NLP-based company name extraction

### 📊 Statistics Display
- Total article count
- Filtered results count
- API status indicator
- Visual stat cards with animations

### 🖼️ Image Handling
- Featured images for each article
- Fallback placeholder images
- Optimized image loading
- Hover effects on images

### 📈 AI-Powered Analysis
- **ML Sentiment Analysis**: Random Forest classifier with TF-IDF features
- **Fallback System**: Keyword-based analysis when ML model fails
- **Trend Prediction**: Upward, downward, or neutral market trend analysis
- **Price Prediction**: Percentage-based price change predictions
- **Confidence Scoring**: AI confidence levels for each prediction
- **Model Training**: 45 financial news samples for specialized accuracy
- **Continuous Learning**: Automatic model retraining with new data
- **Data Storage**: Excel-based data management for scraped articles

## Customization

### Styling
The app uses CSS custom properties and can be easily customized by modifying:
- `frontend/src/App.css` - Main component styles
- `frontend/src/index.css` - Global styles

### Backend Configuration
Modify `app.py` to:
- Change the news source URL
- Adjust the number of articles fetched
- Add new scraping logic
- Modify API response format

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure the Flask backend is running and CORS is enabled
2. **No Images**: Check if the image URLs are accessible, fallback images will be shown
3. **Search Not Working**: Verify that the search functionality is properly connected to the news data
4. **Slow Loading**: The app scrapes real-time data, so loading times may vary

### Development Tips

- Use browser developer tools to debug API calls
- Check the browser console for any JavaScript errors
- Monitor the Flask server logs for backend issues
- Test on different devices to ensure responsive design

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section
2. Review the browser console for errors
3. Ensure all dependencies are properly installed
4. Verify both backend and frontend are running

---

**Happy Trading! 📈💰** 