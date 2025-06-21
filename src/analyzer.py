import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import warnings

from .utils import (
    ensure_directories_exist, save_json, format_currency, 
    clean_text, get_timestamp, send_webhook_notification,
    estimate_api_cost
)
from config.settings import settings

warnings.filterwarnings('ignore')

class YahooFinanceNewsAnalyzer:
    """
    Complete system for collecting Yahoo Finance news and analyzing with ChatGPT
    Includes news collection, technical analysis, and AI-powered insights
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the analyzer with OpenAI API key
        
        Args:
            openai_api_key: OpenAI API key for ChatGPT analysis
        """
        # Setup OpenAI client
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Ensure required directories exist
        ensure_directories_exist([
            settings.DATA_DIR,
            settings.ANALYSIS_DIR,
            settings.CHARTS_DIR
        ])
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logging.info("Yahoo Finance News Analyzer initialized successfully")
    
    def get_yahoo_finance_data(self, ticker: str, period: str = None) -> Dict:
        """
        Collect comprehensive data from Yahoo Finance for a single ticker
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for historical data
            
        Returns:
            Dictionary containing news, price data, and company info
        """
        period = period or settings.HISTORICAL_PERIOD
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get news data
            news = stock.news
            processed_news = []
            
            for article in news:
                news_item = {
                    'title': clean_text(article.get('title', '')),
                    'summary': clean_text(article.get('summary', '')),
                    'link': article.get('link', ''),
                    'published_date': datetime.fromtimestamp(
                        article.get('providerPublishTime', time.time())
                    ).isoformat(),
                    'publisher': article.get('publisher', ''),
                    'uuid': article.get('uuid', ''),
                }
                processed_news.append(news_item)
            
            # Get historical price data
            hist = stock.history(period=period)
            
            # Get company info
            info = stock.info
            
            # Calculate technical indicators
            technical_data = self.calculate_technical_indicators(hist)
            
            return {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': hist['Close'].iloc[-1] if not hist.empty else 0,
                'previous_close': info.get('previousClose', 0),
                'news': processed_news,
                'price_history': hist.to_dict('records') if not hist.empty else [],
                'technical_indicators': technical_data,
                'company_info': {
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'price_to_book': info.get('priceToBook'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'roe': info.get('returnOnEquity'),
                    'profit_margin': info.get('profitMargins'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth')
                },
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error collecting data for {ticker}: {str(e)}")
            return {'ticker': ticker, 'error': str(e)}
    
    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict:
        """
        Calculate various technical indicators from price data
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing technical indicators
        """
        if price_data.empty:
            return {}
        
        try:
            # Create a copy to avoid modifying original data
            df = price_data.copy()
            
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Current values (using last available data)
            current_price = df['Close'].iloc[-1]
            
            return {
                'current_price': float(current_price),
                'sma_20': float(df['SMA_20'].iloc[-1]) if not pd.isna(df['SMA_20'].iloc[-1]) else None,
                'sma_50': float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else None,
                'sma_200': float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else None,
                'rsi': float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None,
                'macd': float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else None,
                'macd_signal': float(df['MACD_Signal'].iloc[-1]) if not pd.isna(df['MACD_Signal'].iloc[-1]) else None,
                'bb_upper': float(df['BB_Upper'].iloc[-1]) if not pd.isna(df['BB_Upper'].iloc[-1]) else None,
                'bb_lower': float(df['BB_Lower'].iloc[-1]) if not pd.isna(df['BB_Lower'].iloc[-1]) else None,
                'bb_position': self._calculate_bb_position(current_price, df['BB_Upper'].iloc[-1], df['BB_Lower'].iloc[-1]),
                'price_vs_sma20': self._calculate_price_vs_ma(current_price, df['SMA_20'].iloc[-1]),
                'price_vs_sma50': self._calculate_price_vs_ma(current_price, df['SMA_50'].iloc[-1]),
                'volume_avg': float(df['Volume'].tail(20).mean()),
                'volume_current': float(df['Volume'].iloc[-1]),
                'volatility': float(df['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100)
            }
            
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    def _calculate_bb_position(self, price: float, bb_upper: float, bb_lower: float) -> Optional[float]:
        """Calculate Bollinger Band position (0-1 scale)"""
        try:
            if pd.isna(bb_upper) or pd.isna(bb_lower) or bb_upper == bb_lower:
                return None
            return (price - bb_lower) / (bb_upper - bb_lower)
        except:
            return None
    
    def _calculate_price_vs_ma(self, price: float, ma_value: float) -> Optional[float]:
        """Calculate price vs moving average percentage"""
        try:
            if pd.isna(ma_value) or ma_value == 0:
                return None
            return ((price - ma_value) / ma_value) * 100
        except:
            return None
    
    def create_stock_chart(self, ticker: str, price_data: pd.DataFrame) -> str:
        """
        Create technical analysis chart for the stock
        
        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with price and indicator data
            
        Returns:
            Path to saved chart image
        """
        try:
            # Add technical indicators to the dataframe
            df = self.add_indicators_to_dataframe(price_data.copy())
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{ticker} Technical Analysis', fontsize=16, fontweight='bold')
            
            # Price and Moving Averages
            ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='black')
            if 'SMA_20' in df.columns and not df['SMA_20'].isna().all():
                ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7, color='blue')
            if 'SMA_50' in df.columns and not df['SMA_50'].isna().all():
                ax1.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7, color='red')
            ax1.set_title('Price and Moving Averages')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bollinger Bands
            if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                ax2.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='black')
                ax2.plot(df.index, df['BB_Upper'], label='BB Upper', alpha=0.7, color='red')
                ax2.plot(df.index, df['BB_Lower'], label='BB Lower', alpha=0.7, color='green')
                ax2.plot(df.index, df['BB_Middle'], label='BB Middle', alpha=0.5, color='blue')
                ax2.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')
            else:
                ax2.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='black')
            ax2.set_title('Bollinger Bands')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # RSI
            if 'RSI' in df.columns and not df['RSI'].isna().all():
                ax3.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
                ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                ax3.set_ylim(0, 100)
            ax3.set_title('RSI (Relative Strength Index)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # MACD
            if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                ax4.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=2)
                ax4.plot(df.index, df['MACD_Signal'], label='Signal', color='red', linewidth=2)
                ax4.bar(df.index, df['MACD_Histogram'], label='Histogram', alpha=0.3, color='green')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title('MACD')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f'{settings.CHARTS_DIR}/{ticker}_technical_analysis_{get_timestamp()}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Chart saved: {chart_path}")
            return chart_path
            
        except Exception as e:
            logging.error(f"Error creating chart for {ticker}: {str(e)}")
            return ""
    
    def add_indicators_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price dataframe for charting
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        try:
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            return df
            
        except Exception as e:
            logging.error(f"Error adding indicators: {str(e)}")
            return df
    
    def analyze_with_chatgpt(self, stock_data: Dict) -> Dict:
        """
        Analyze stock data using ChatGPT API
        
        Args:
            stock_data: Complete stock data including news and technical indicators
            
        Returns:
            Dictionary containing AI analysis results
        """
        try:
            # Prepare data for analysis
            ticker = stock_data.get('ticker', 'Unknown')
            company_name = stock_data.get('company_name', 'Unknown')
            news_items = stock_data.get('news', [])
            technical_data = stock_data.get('technical_indicators', {})
            company_info = stock_data.get('company_info', {})
            
            # Create comprehensive prompt
            news_text = "\n".join([
                f"- {item.get('title', '')}: {item.get('summary', '')[:200]}..."
                for item in news_items[:10]  # Limit to top 10 news items
            ])
            
            technical_summary = self._format_technical_summary(technical_data)
            fundamental_summary = self._format_fundamental_summary(company_info)
            
            prompt = f"""
As a professional financial analyst, please provide a comprehensive analysis of {company_name} ({ticker}) based on the following data:

RECENT NEWS:
{news_text}

{technical_summary}

{fundamental_summary}

Please provide analysis in the following format:

1. NEWS SUMMARY:
Summarize the key themes and sentiment from recent news

2. MARKET SENTIMENT:
Overall market sentiment towards this stock based on news and social indicators

3. TECHNICAL ANALYSIS:
Analysis of technical indicators and chart patterns

4. FUNDAMENTAL ANALYSIS:
Assessment of company's financial health and valuation

5. RISK ASSESSMENT:
Key risks and concerns for this investment

6. INVESTMENT RECOMMENDATION:
Clear recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell) with target price range and time horizon

7. KEY CATALYSTS:
Upcoming events or factors that could significantly impact the stock price

Please be specific, data-driven, and provide actionable insights. Use professional financial terminology but keep explanations clear.
"""
            
            # Call ChatGPT API
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior financial analyst with 15+ years of experience in equity research and technical analysis. Provide detailed, professional investment analysis."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse the analysis into structured format
            analysis_sections = self.parse_analysis_response(analysis_text)
            
            return {
                'ticker': ticker,
                'company_name': company_name,
                'analysis_date': datetime.now().isoformat(),
                'raw_analysis': analysis_text,
                'structured_analysis': analysis_sections,
                'tokens_used': response.usage.total_tokens,
                'model_used': settings.OPENAI_MODEL
            }
            
        except Exception as e:
            logging.error(f"Error in ChatGPT analysis for {stock_data.get('ticker')}: {str(e)}")
            return {
                'ticker': stock_data.get('ticker'),
                'error': str(e),
                'analysis_date': datetime.now().isoformat()
            }
    
    def _format_technical_summary(self, technical_data: Dict) -> str:
        """Format technical indicators for prompt"""
        if not technical_data:
            return "Technical Indicators: No data available"
        
        return f"""
Technical Indicators:
- Current Price: ${technical_data.get('current_price', 0):.2f}
- RSI: {technical_data.get('rsi', 'N/A')}
- Price vs SMA20: {technical_data.get('price_vs_sma20', 'N/A')}%
- Price vs SMA50: {technical_data.get('price_vs_sma50', 'N/A')}%
- MACD: {technical_data.get('macd', 'N/A')}
- Bollinger Band Position: {technical_data.get('bb_position', 'N/A')}
- Volatility: {technical_data.get('volatility', 'N/A')}%
"""
    
    def _format_fundamental_summary(self, company_info: Dict) -> str:
        """Format fundamental data for prompt"""
        if not company_info:
            return "Fundamental Data: No data available"
        
        return f"""
Fundamental Data:
- P/E Ratio: {company_info.get('pe_ratio', 'N/A')}
- Forward P/E: {company_info.get('forward_pe', 'N/A')}
- Price to Book: {company_info.get('price_to_book', 'N/A')}
- ROE: {company_info.get('roe', 'N/A')}
- Profit Margin: {company_info.get('profit_margin', 'N/A')}
- Revenue Growth: {company_info.get('revenue_growth', 'N/A')}
"""
    
    def parse_analysis_response(self, analysis_text: str) -> Dict:
        """
        Parse ChatGPT response into structured sections
        
        Args:
            analysis_text: Raw analysis text from ChatGPT
            
        Returns:
            Dictionary with structured analysis sections
        """
        sections = {
            'news_summary': '',
            'market_sentiment': '',
            'technical_analysis': '',
            'fundamental_analysis': '',
            'risk_assessment': '',
            'investment_recommendation': '',
            'key_catalysts': ''
        }
        
        try:
            # Split by numbered sections
            lines = analysis_text.split('\n')
            current_section = None
            current_content = []
            
            section_keywords = {
                'news summary': 'news_summary',
                'market sentiment': 'market_sentiment',
                'technical analysis': 'technical_analysis',
                'fundamental analysis': 'fundamental_analysis',
                'risk assessment': 'risk_assessment',
                'investment recommendation': 'investment_recommendation',
                'key catalysts': 'key_catalysts'
            }
            
            for line in lines:
                line = line.strip()
                
                # Check if line starts a new section
                section_found = False
                for keyword, section_key in section_keywords.items():
                    if keyword.lower() in line.lower() and ':' in line:
                        if current_section and current_content:
                            sections[current_section] = '\n'.join(current_content).strip()
                        current_section = section_key
                        current_content = []
                        section_found = True
                        break
                
                if not section_found and current_section:
                    current_content.append(line)
            
            # Add the last section
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
                
        except Exception as e:
            logging.error(f"Error parsing analysis response: {str(e)}")
            sections['raw_text'] = analysis_text
        
        return sections
    
    def generate_portfolio_analysis(self, all_analyses: List[Dict]) -> Dict:
        """
        Generate overall portfolio/market analysis from individual stock analyses
        
        Args:
            all_analyses: List of individual stock analyses
            
        Returns:
            Portfolio-level analysis
        """
        try:
            # Prepare summary data
            portfolio_summary = []
            for analysis in all_analyses:
                if 'error' not in analysis:
                    portfolio_summary.append(f"""
{analysis['ticker']} ({analysis['company_name']}):
Investment Recommendation: {analysis['structured_analysis'].get('investment_recommendation', 'N/A')[:200]}
Key Risks: {analysis['structured_analysis'].get('risk_assessment', 'N/A')[:200]}
""")
            
            portfolio_text = "\n".join(portfolio_summary)
            
            prompt = f"""
Based on the following individual stock analyses, provide a comprehensive market and portfolio overview:

{portfolio_text}

Please provide:

1. OVERALL MARKET SENTIMENT:
What is the general market sentiment based on these stocks?

2. SECTOR TRENDS:
Are there any sector-specific trends or themes emerging?

3. PORTFOLIO DIVERSIFICATION:
Comments on portfolio balance and diversification

4. MARKET RISKS:
Key market-wide risks to monitor

5. INVESTMENT STRATEGY:
Recommended overall investment strategy and allocation

6. MARKET OUTLOOK:
Short-term (1-3 months) and medium-term (6-12 months) market outlook

Keep the analysis concise but comprehensive.
"""
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior portfolio manager providing market-wide analysis and investment strategy recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=settings.TEMPERATURE
            )
            
            return {
                'analysis_date': datetime.now().isoformat(),
                'portfolio_analysis': response.choices[0].message.content,
                'stocks_analyzed': len(all_analyses),
                'tokens_used': response.usage.total_tokens
            }
            
        except Exception as e:
            logging.error(f"Error generating portfolio analysis: {str(e)}")
            return {'error': str(e)}
    
    def run_complete_analysis(self, tickers: List[str]) -> Dict:
        """
        Run complete analysis pipeline for multiple stocks
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Complete analysis results
        """
        logging.info(f"Starting complete analysis for tickers: {tickers}")
        
        all_stock_data = []
        all_analyses = []
        
        # Show cost estimation
        cost_estimate = estimate_api_cost(len(tickers))
        logging.info(f"Estimated API cost: ${cost_estimate['estimated_cost']:.2f}")
        
        # Collect data and analyze each stock
        for i, ticker in enumerate(tickers, 1):
            logging.info(f"Processing {ticker} ({i}/{len(tickers)})...")
            
            # Get Yahoo Finance data
            stock_data = self.get_yahoo_finance_data(ticker)
            
            if 'error' not in stock_data:
                all_stock_data.append(stock_data)
                
                # Create technical chart
                if stock_data.get('price_history'):
                    price_df = pd.DataFrame(stock_data['price_history'])
                    if not price_df.empty:
                        # Set proper datetime index
                        if 'Date' not in price_df.columns:
                            price_df.reset_index(inplace=True)
                        
                        if 'Date' in price_df.columns:
                            price_df['Date'] = pd.to_datetime(price_df['Date'])
                            price_df.set_index('Date', inplace=True)
                        
                        chart_path = self.create_stock_chart(ticker, price_df)
                        stock_data['chart_path'] = chart_path
                
                # Analyze with ChatGPT
                analysis = self.analyze_with_chatgpt(stock_data)
                all_analyses.append(analysis)
                
                # Rate limiting delay
                time.sleep(settings.REQUEST_DELAY)
            else:
                logging.error(f"Failed to collect data for {ticker}")
        
        # Generate portfolio analysis
        portfolio_analysis = self.generate_portfolio_analysis(all_analyses)
        
        # Compile final results
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'tickers_analyzed': tickers,
            'individual_analyses': all_analyses,
            'portfolio_analysis': portfolio_analysis,
            'raw_stock_data': all_stock_data,
            'cost_estimate': cost_estimate,
            'summary': {
                'total_stocks': len(tickers),
                'successful_analyses': len(all_analyses),
                'failed_analyses': len(tickers) - len(all_analyses)
            }
        }
        
        # Save results
        self.save_analysis_results(results)
        
        logging.info("Complete analysis finished successfully")
        return results
    
    def save_analysis_results(self, results: Dict) -> str:
        """
        Save analysis results to files
        
        Args:
            results: Complete analysis results
            
        Returns:
            Path to saved file
        """
        try:
            timestamp = get_timestamp()
            
            # Save complete results as JSON
            json_filename = f"{settings.ANALYSIS_DIR}/complete_analysis_{timestamp}.json"
            save_json(results, json_filename)
            
            # Save summary as readable text
            text_filename = f"{settings.ANALYSIS_DIR}/analysis_summary_{timestamp}.txt"
            self._save_text_summary(results, text_filename)
            
            # Create HTML dashboard
            html_filename = self._create_html_dashboard(results, timestamp)
            
            logging.info(f"Analysis results saved to {json_filename}, {text_filename}, and {html_filename}")
            return json_filename
            
        except Exception as e:
            logging.error(f"Error saving analysis results: {str(e)}")
            return ""
    
    def _save_text_summary(self, results: Dict, filename: str) -> None:
        """Save analysis results as readable text"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"STOCK ANALYSIS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Summary
                f.write(f"ANALYSIS SUMMARY:\n")
                f.write(f"- Total Stocks: {results['summary']['total_stocks']}\n")
                f.write(f"- Successful Analyses: {results['summary']['successful_analyses']}\n")
                f.write(f"- Failed Analyses: {results['summary']['failed_analyses']}\n")
                f.write(f"- Estimated Cost: ${results['cost_estimate']['estimated_cost']:.2f}\n\n")
                
                # Individual analyses
                for analysis in results['individual_analyses']:
                    if 'error' not in analysis:
                        f.write(f"TICKER: {analysis['ticker']} - {analysis['company_name']}\n")
                        f.write("-" * 60 + "\n")
                        f.write(analysis.get('raw_analysis', 'No analysis available'))
                        f.write("\n\n" + "=" * 80 + "\n\n")
                
                # Portfolio analysis
                if 'portfolio_analysis' in results and 'error' not in results['portfolio_analysis']:
                    f.write("PORTFOLIO & MARKET ANALYSIS\n")
                    f.write("=" * 80 + "\n")
                    f.write(results['portfolio_analysis'].get('portfolio_analysis', ''))
        except Exception as e:
            logging.error(f"Error saving text summary: {str(e)}")
    
    def _create_html_dashboard(self, results: Dict, timestamp: str) -> str:
        """Create HTML dashboard"""
        try:
            html_filename = f"{settings.ANALYSIS_DIR}/dashboard_{timestamp}.html"
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stock Analysis Dashboard</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px; 
            border-radius: 10px; 
            margin-bottom: 30px;
            text-align: center;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stock-card {{ 
            background: white;
            margin: 20px 0; 
            padding: 25px; 
            border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .stock-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
        }}
        .ticker {{ 
            font-size: 24px; 
            font-weight: bold; 
            color: #2c5aa0; 
        }}
        .recommendation {{ 
            font-weight: bold; 
            font-size: 16px; 
            padding: 8px 16px;
            border-radius: 20px;
            color: white;
        }}
        .buy {{ background-color: #28a745; }}
        .hold {{ background-color: #ffc107; color: #212529; }}
        .sell {{ background-color: #dc3545; }}
        .section {{ margin: 20px 0; }}
        .section h3 {{ 
            color: #333; 
            border-left: 4px solid #667eea;
            padding-left: 15px;
            margin-bottom: 10px;
        }}
        .portfolio-section {{ 
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px; 
            margin: 30px 0; 
            border-radius: 10px; 
        }}
        pre {{ 
            background-color: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 5px; 
            white-space: pre-wrap; 
            line-height: 1.6;
        }}
        .metric {{ text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Stock Analysis Dashboard</h1>
            <p><strong>Generated:</strong> {results['analysis_timestamp'][:19]}</p>
            <p><strong>Stocks Analyzed:</strong> {', '.join(results['tickers_analyzed'])}</p>
        </div>
        
        <div class="summary-cards">
            <div class="card metric">
                <div class="metric-value">{results['summary']['total_stocks']}</div>
                <div class="metric-label">Total Stocks</div>
            </div>
            <div class="card metric">
                <div class="metric-value">{results['summary']['successful_analyses']}</div>
                <div class="metric-label">Successful Analyses</div>
            </div>
            <div class="card metric">
                <div class="metric-value">${results['cost_estimate']['estimated_cost']:.2f}</div>
                <div class="metric-label">Estimated Cost</div>
            </div>
        </div>
"""
            
            # Individual stock analyses
            for analysis in results['individual_analyses']:
                if 'error' not in analysis:
                    structured = analysis['structured_analysis']
                    
                    # Determine recommendation class
                    rec_text = structured.get('investment_recommendation', '').lower()
                    rec_class = 'hold'  # default
                    if 'buy' in rec_text and 'strong' in rec_text:
                        rec_class = 'buy'
                    elif 'buy' in rec_text:
                        rec_class = 'buy'
                    elif 'sell' in rec_text:
                        rec_class = 'sell'
                    
                    html_content += f"""
        <div class="stock-card">
            <div class="stock-header">
                <div>
                    <div class="ticker">{analysis['ticker']}</div>
                    <div style="color: #666;">{analysis['company_name']}</div>
                </div>
                <div class="recommendation {rec_class}">
                    {structured.get('investment_recommendation', 'N/A').split('.')[0]}
                </div>
            </div>
            
            <div class="section">
                <h3>📰 News Summary</h3>
                <p>{structured.get('news_summary', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h3>📈 Technical Analysis</h3>
                <p>{structured.get('technical_analysis', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h3>⚠️ Risk Assessment</h3>
                <p>{structured.get('risk_assessment', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h3>🎯 Key Catalysts</h3>
                <p>{structured.get('key_catalysts', 'N/A')}</p>
            </div>
        </div>
"""
            
            # Portfolio analysis
            if 'portfolio_analysis' in results and 'error' not in results['portfolio_analysis']:
                html_content += f"""
        <div class="portfolio-section">
            <h2>📊 Portfolio & Market Analysis</h2>
            <pre>{results['portfolio_analysis'].get('portfolio_analysis', 'N/A')}</pre>
        </div>
"""
            
            html_content += """
    </div>
</body>
</html>
"""
            
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return html_filename
            
        except Exception as e:
            logging.error(f"Error creating HTML dashboard: {str(e)}")
            return ""
