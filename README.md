# 📈 AI Stock Predictor

An intelligent AI-powered stock prediction system for Indian stocks (Nifty 50) using deep learning and ensemble machine learning models. Predicts intraday price movements with technical analysis, sentiment analysis, and sector strength indicators.

## 🎯 Features

- **AI-Powered Predictions**: Ensemble of Transformer neural networks and XGBoost classifiers
- **Real-Time Dashboard**: Interactive Streamlit UI with live market data visualization
- **Technical Analysis**: Support/resistance levels, breakout detection, moving averages
- **Sentiment Analysis**: Market and stock-specific news sentiment scoring
- **Sector Analysis**: Real-time sector strength indicators
- **Smart Recommendations**: Automated stock picker based on multiple signals
- **Feature Engineering**: Intelligent feature selection from 22 technical indicators
- **Nifty 50 Coverage**: Comprehensive analysis across all NSE top 50 stocks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/piyushnitkkr/ai-stock-predictor.git
cd ai-stock-predictor

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# Start the Streamlit app
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to access the dashboard.

## 📊 How It Works

### 1. Data Collection
- Fetches historical OHLCV data for Nifty 50 stocks
- Downloads market indices (NIFTY50 benchmark)
- Collects real-time news data for sentiment analysis

### 2. Feature Engineering
- Generates 22+ technical indicators:
  - Moving averages (SMA, EMA)
  - Momentum indicators (RSI, MACD, Stochastic)
  - Volatility measures (Bollinger Bands, ATR)
  - Price action patterns

### 3. Feature Selection
- Uses ensemble approach:
  - Mutual Information (40% weight)
  - F-statistic classifier (30% weight)
  - Random Forest importance (30% weight)
- Selects top 10 features for model training

### 4. Model Training
- **Transformer Model**: Captures temporal patterns in price sequences (120-day window)
- **XGBoost Classifier**: Non-linear relationship learning
- **Ensemble Voting**: Combines predictions for better accuracy

### 5. Prediction
- Classifies next day movement into:
  - 🟢 **Bullish** (>1.5% gain)
  - 🔴 **Bearish** (<-1.5% loss)
  - 🟡 **Sideways** (±1.5%)

## 📈 Model Performance

- **Accuracy**: 70%
- **Sequence Length**: 120 days
- **Class Labels**: Bullish, Bearish, Sideways

## 🗂️ Project Structure

```
ai-stock-predictor/
├── app.py                      # Main Streamlit dashboard
├── predict.py                  # Prediction engine
├── train_model.py              # Model training pipeline
├── data_fetcher.py             # Downloads stock & market data
├── features.py                 # Technical indicator calculation
├── feature_selection.save      # Serialized feature selector
├── model_transformer.py        # Transformer model architecture
├── model_xgb.py                # XGBoost configuration
├── scaler.save                 # Data normalization scaler
├── support_resistance.py        # Support/resistance level detection
├── breakout.py                 # Breakout pattern detection
├── sector_analysis.py          # Sector strength calculation
├── news_sentiment.py           # News sentiment analysis
├── stock_recommender.py        # Automated stock recommendation
├── nifty50.py                  # Nifty 50 ticker list
├── transformer_model.keras     # Trained Transformer model
├── xgb_model.save              # Trained XGBoost model
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📝 Usage Examples

### Using the Dashboard

1. **Stock Analysis**:
   - Enter a stock ticker (e.g., RELIANCE, TCS, INFY)
   - View real-time price with AI prediction
   - Analyze technical levels and patterns

2. **Market Overview**:
   - Check sector performance
   - View top recommendations
   - Read latest market news with sentiment

3. **Technical Analysis**:
   - Support and resistance levels
   - Breakout signals
   - Moving average trends

### Using the Prediction Engine Programmatically

```python
from predict import predict_stock
from data_fetcher import get_stock_data

# Fetch data
data = get_stock_data("RELIANCE")

# Get prediction
result = predict_stock("RELIANCE", data)

print(f"Bullish: {result['bullish']:.1%}")
print(f"Bearish: {result['bearish']:.1%}")
print(f"Sideways: {result['sideways']:.1%}")
```

## 🔄 Training the Model

```bash
# Retrain the entire model on latest data
python train_model.py
```

This will:
1. Download latest data for all Nifty 50 stocks
2. Generate features and select top 10
3. Train Transformer and XGBoost models
4. Save trained models and scalers
5. Evaluate on validation set

## 🛠️ Configuration

Key hyperparameters in `train_model.py`:

```python
SEQUENCE_LENGTH = 120      # Days of historical data per sequence
LABEL_THRESHOLD = 0.015    # 1.5% threshold for bullish/bearish
TOP_FEATURES = 10          # Number of features to select
```

## 📦 Dependencies

See `requirements.txt` for complete list:
- **yfinance**: Stock data fetching
- **tensorflow/keras**: Transformer model
- **xgboost**: Gradient boosting
- **pandas/numpy**: Data manipulation
- **scikit-learn**: Feature selection & scaling
- **streamlit**: Web dashboard
- **plotly**: Interactive charts
- **newsapi**: News data
- **textblob**: Sentiment analysis

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Stock market predictions involve inherent risks. Past performance does not guarantee future results. Always conduct your own research and consult with financial advisors before making investment decisions.

## 📧 Contact & Contributions

- **Author**: Piyush Sharma
- **Repository**: https://github.com/piyushnitkkr/ai-stock-predictor

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated**: March 18, 2026
