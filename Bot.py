import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext, JobQueue
import talib
import datetime
import os
import time
from io import BytesIO
import yfinance as yf
import pandas_ta as ta
from textblob import TextBlob
import nltk
from flask import Flask
import threading

# Create Flask app for health checks
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Nifty/BankNifty Options Bot is running!", 200

# Download NLTK data for TextBlob
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Configuration
BOT_TOKEN = os.environ.get('BOT_TOKEN')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')  # Get from newsapi.org
SCAN_INTERVAL = 900  # 15 minutes for auto scans

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class OptionScanner:
    def __init__(self):
        self.indices = {
            "NIFTY": {"symbol": "^NSEI", "name": "Nifty 50", "lot_size": 50},
            "BANKNIFTY": {"symbol": "^NSEBANK", "name": "Bank Nifty", "lot_size": 25}
        }
        self.global_indices = {
            "US": {"symbol": "^GSPC", "name": "S&P 500"},
            "EUROPE": {"symbol": "^STOXX50E", "name": "Euro Stoxx 50"},
            "ASIA": {"symbol": "^HSI", "name": "Hang Seng"}
        }
        self.sentiment_data = {}
        self.last_updated = None

    def fetch_index_data(self, index_symbol):
        """Fetch real-time data for an index"""
        try:
            index = yf.Ticker(index_symbol)
            df = index.history(period='2d', interval='15m')
            if df.empty or len(df) < 20:
                return None
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.error(f"Error fetching {index_symbol}: {e}")
            return None

    def calculate_technical(self, df):
        """Calculate technical indicators"""
        try:
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Price indicators
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)
            df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['close'])
            
            # Volatility
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Trend
            df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Support/Resistance
            df['SUPERT'] = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)['SUPERT_7_3.0']
            
            return df
        except Exception as e:
            logger.error(f"Technical calculation error: {e}")
            return df

    def get_news_sentiment(self):
        """Fetch market news and analyze sentiment"""
        try:
            # Fetch financial news
            news_url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}"
            response = requests.get(news_url)
            news_data = response.json()
            
            # Analyze sentiment
            sentiment_scores = []
            for article in news_data.get('articles', [])[:10]:
                analysis = TextBlob(article['title'] + " " + (article['description'] or ""))
                sentiment_scores.append(analysis.sentiment.polarity)
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            return avg_sentiment, [article['title'] for article in news_data.get('articles', [])[:3]]
        except Exception as e:
            logger.error(f"News error: {e}")
            return 0, []

    def get_global_market_status(self):
        """Check pre-market status of global indices"""
        status = {}
        for region, data in self.global_indices.items():
            try:
                index = yf.Ticker(data['symbol'])
                hist = index.history(period='2d')
                if not hist.empty:
                    last_close = hist['Close'].iloc[-1]
                    current = index.fast_info.last_price
                    change_pct = ((current - last_close) / last_close) * 100
                    status[region] = {
                        "name": data['name'],
                        "change": f"{change_pct:.2f}%",
                        "direction": "UP" if change_pct > 0 else "DOWN"
                    }
            except:
                continue
        return status

    def generate_signal(self, df, index_name):
        """Generate options signal for an index"""
        if df is None or len(df) < 20:
            return None
        
        latest = df.iloc[-1]
        
        # Skip if missing values
        if any(pd.isna(latest.get(col, None)) for col in ['RSI', 'MACD', 'MACD_Signal', 'ATR', 'SUPERT']):
            return None
        
        # Calculate signal strength
        rsi_position = latest['RSI']
        macd_cross = 1 if latest['MACD'] > latest['MACD_Signal'] else -1
        super_trend = 1 if latest['close'] > latest['SUPERT'] else -1
        
        # Calculate signal score
        signal_score = (70 - min(rsi_position, 70)) * macd_cross * super_trend
        
        # Determine signal type
        if signal_score > 20:
            signal_type = "CALL"
            reason = "Bullish Technical Setup"
        elif signal_score < -20:
            signal_type = "PUT"
            reason = "Bearish Technical Setup"
        else:
            return None
        
        # Calculate levels
        atr = latest['ATR']
        entry = latest['close']
        
        if signal_type == "CALL":
            stop_loss = max(df['low'].tail(3).min(), entry - atr * 1.5)
            target = entry + atr * 2.5
        else:
            stop_loss = min(df['high'].tail(3).max(), entry + atr * 1.5)
            target = entry - atr * 2.5
        
        # Risk management
        risk_per_point = abs(entry - stop_loss)
        reward_per_point = abs(target - entry)
        risk_reward = reward_per_point / risk_per_point if risk_per_point > 0 else 0
        
        return {
            "index": index_name,
            "type": signal_type,
            "reason": reason,
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "risk_reward": round(risk_reward, 2),
            "signal_score": round(signal_score, 2),
            "current": round(entry, 2),
            "timestamp": datetime.datetime.now()
        }

    def scan_market(self):
        """Scan indices and generate signals with sentiment"""
        logger.info("Starting market scan...")
        signals = []
        
        # Get market sentiment
        sentiment, news_headlines = self.get_news_sentiment()
        global_markets = self.get_global_market_status()
        self.sentiment_data = {
            "sentiment": sentiment,
            "news": news_headlines,
            "global_markets": global_markets
        }
        
        for index_name, data in self.indices.items():
            try:
                df = self.fetch_index_data(data['symbol'])
                if df is None:
                    continue
                
                df = self.calculate_technical(df)
                signal = self.generate_signal(df, index_name)
                
                if signal:
                    # Add additional data
                    signal['lot_size'] = data['lot_size']
                    signal['symbol'] = data['symbol']
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error scanning {index_name}: {e}")
        
        # Sort by signal strength
        signals.sort(key=lambda x: abs(x['signal_score']), reverse=True)
        
        self.last_updated = datetime.datetime.now()
        logger.info(f"Market scan completed. Found {len(signals)} signals.")
        return signals

    def get_best_signal(self):
        """Get the best signal from recent scan"""
        if not self.last_updated or (datetime.datetime.now() - self.last_updated).seconds > 600:
            return self.scan_market()
        return self.scan_market()

    def format_signal(self, signal):
        """Format signal for Telegram"""
        index_data = self.indices.get(signal['index'], {})
        lot_size = signal.get('lot_size', 50)
        point_diff = abs(signal['entry'] - signal['target'])
        risk_per_lot = abs(signal['entry'] - signal['stop_loss']) * lot_size
        reward_per_lot = point_diff * lot_size
        
        # Format global markets status
        global_status = "\n".join(
            [f"{data['name']}: {data['change']} ({data['direction']})" 
             for region, data in self.sentiment_data['global_markets'].items()]
        ) if 'global_markets' in self.sentiment_data else "Global data unavailable"
        
        # Format news headlines
        news_headlines = "\n- ".join(self.sentiment_data.get('news',