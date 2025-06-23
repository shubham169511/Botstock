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
import feedparser
import pandas_ta as ta
from textblob import TextBlob

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
        )
        
        # Format news headlines
        news_headlines = "\n- ".join(self.sentiment_data['news'])
        
        # Sentiment interpretation
        sentiment = self.sentiment_data['sentiment']
        sentiment_label = "Bullish" if sentiment > 0.1 else "Bearish" if sentiment < -0.1 else "Neutral"
        
        return (
            f"📈 {index_data.get('name', signal['index'])} OPTION SIGNAL\n"
            f"🏷️ Type: {signal['type']}\n"
            f"📌 Reason: {signal['reason']}\n"
            f"📊 Signal Strength: {abs(signal['signal_score']):.0f}/100\n\n"
            f"💵 Current: {signal['current']}\n"
            f"🎯 Entry: {signal['entry']}\n"
            f"🛑 Stop Loss: {signal['stop_loss']}\n"
            f"🎯 Target: {signal['target']}\n\n"
            f"💰 Profit Per Lot: ₹{reward_per_lot:.0f} ({point_diff:.1f} points)\n"
            f"⚠️ Risk Per Lot: ₹{risk_per_lot:.0f}\n"
            f"⚖️ Risk/Reward: 1:{signal['risk_reward']:.1f}\n\n"
            f"🌍 GLOBAL MARKETS:\n{global_status}\n\n"
            f"📰 MARKET SENTIMENT: {sentiment_label.upper()} ({sentiment:.2f})\n"
            f"📌 TOP NEWS:\n- {news_headlines}\n\n"
            f"⏰ Updated: {self.last_updated.strftime('%Y-%m-%d %H:%M')}\n"
            f"⚠️ Disclaimer: Trade responsibly. Set stop losses."
        )

    def plot_signal(self, signal):
        """Generate chart for the signal"""
        try:
            df = yf.Ticker(signal['symbol']).history(period='2d', interval='15m')
            
            plt.figure(figsize=(12, 8))
            
            # Price chart
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df['Close'], label='Price', color='blue')
            plt.title(f"{self.indices[signal['index']]['name']} - Options Signal")
            
            # Plot signal markers
            if signal['type'] == "CALL":
                plt.plot(df.index[-1], signal['entry'], 'g^', markersize=12, label='Entry')
                plt.axhline(y=signal['stop_loss'], color='r', linestyle='--', label='Stop Loss')
                plt.axhline(y=signal['target'], color='g', linestyle='--', label='Target')
            else:
                plt.plot(df.index[-1], signal['entry'], 'rv', markersize=12, label='Entry')
                plt.axhline(y=signal['stop_loss'], color='r', linestyle='--', label='Stop Loss')
                plt.axhline(y=signal['target'], color='g', linestyle='--', label='Target')
            
            plt.grid(True)
            plt.legend()
            
            # Indicators
            plt.subplot(2, 1, 2)
            plt.plot(df.index, ta.rsi(df['Close'], length=14), label='RSI', color='purple')
            plt.axhline(y=30, color='green', linestyle='--')
            plt.axhline(y=70, color='red', linestyle='--')
            plt.ylim(0, 100)
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            
            # Save to bytes buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close()
            return buf
        except Exception as e:
            logger.error(f"Chart error: {e}")
            return None

# Initialize scanner
scanner = OptionScanner()

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        "🚀 Nifty/BankNifty Options Bot\n\n"
        "I provide CALL/PUT signals for:\n"
        "✅ Nifty 50\n"
        "✅ Bank Nifty\n\n"
        "Features:\n"
        "- Technical analysis with RSI, MACD, SuperTrend\n"
        "- Global market sentiment\n"
        "- News-based market analysis\n"
        "- Risk-managed trade setups\n\n"
        "Commands:\n"
        "/signal - Get current trading signals\n"
        "/auto - Enable auto-alerts (every 15 min)\n"
        "/stop - Disable auto-alerts\n\n"
        "⚠️ Disclaimer: Not financial advice. Trading involves risk."
    )

def signal_command(update: Update, context: CallbackContext) -> None:
    try:
        update.message.reply_text("🔍 Scanning market for Nifty/BankNifty opportunities...")
        signals = scanner.get_best_signal()
        
        if not signals:
            update.message.reply_text("⏳ No strong signals detected. Market may be consolidating.")
            return
            
        # Send all signals
        for signal in signals:
            chart_buf = scanner.plot_signal(signal)
            message = scanner.format_signal(signal)
            
            if chart_buf:
                update.message.reply_photo(
                    photo=chart_buf,
                    caption=message
                )
                chart_buf.close()
            else:
                update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Signal error: {e}", exc_info=True)
        update.message.reply_text("🔴 Error processing signal. Try again later.")

def auto_alerts(context: CallbackContext):
    job = context.job
    try:
        signals = scanner.get_best_signal()
        
        if signals:
            for signal in signals:
                chart_buf = scanner.plot_signal(signal)
                message = scanner.format_signal(signal)
                
                if chart_buf:
                    context.bot.send_photo(
                        chat_id=job.context,
                        photo=chart_buf,
                        caption=message
                    )
                    chart_buf.close()
                else:
                    context.bot.send_message(
                        chat_id=job.context,
                        text=message
                    )
    except Exception as e:
        logger.error(f"Auto-alert error: {e}")

def enable_auto(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    current_jobs = context.job_queue.get_jobs_by_name(str(chat_id))
    
    if current_jobs:
        update.message.reply_text("ℹ️ Auto-alerts already enabled.")
        return
        
    context.job_queue.run_repeating(
        auto_alerts, 
        interval=SCAN_INTERVAL,
        first=10,
        context=chat_id,
        name=str(chat_id)
    )
    
    update.message.reply_text(f"✅ Auto-alerts enabled! You'll receive signals every {SCAN_INTERVAL//60} minutes.")

def disable_auto(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    current_jobs = context.job_queue.get_jobs_by_name(str(chat_id))
    
    if not current_jobs:
        update.message.reply_text("ℹ️ Auto-alerts are not enabled.")
        return
        
    for job in current_jobs:
        job.schedule_removal()
    update.message.reply_text("🔴 Auto-alerts disabled!")

def error(update: Update, context: CallbackContext):
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():
    TOKEN = BOT_TOKEN
    PORT = int(os.environ.get('PORT', 5000))
    APP_NAME = os.environ.get('APP_NAME')

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    # Command handlers
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("signal", signal_command))
    dp.add_handler(CommandHandler("auto", enable_auto))
    dp.add_handler(CommandHandler("stop", disable_auto))
    dp.add_error_handler(error)

    # Start the Bot
    if APP_NAME:  # Running on Render
        updater.start_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TOKEN,
            webhook_url=f"https://{APP_NAME}.onrender.com/{TOKEN}"
        )
        logger.info(f"Webhook set to: https://{APP_NAME}.onrender.com/{TOKEN}")
    else:  # Running locally
        updater.start_polling()
        logger.info("Bot started in polling mode")
    
    # Initial scan
    scanner.scan_market()
    updater.idle()

if __name__ == '__main__':
    main()