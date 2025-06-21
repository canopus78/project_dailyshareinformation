import logging
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

class MarketFactorAnalyzer:
    def __init__(self, market_factors):
        self.market_factors = market_factors

    def analyze_factors(self):
        # Analyze individual market factors
        fed_policy_score = self.analyze_fed_policy()
        inflation_score = self.analyze_inflation()
        geopolitical_risk_score = self.analyze_geopolitical_risk()
        earnings_growth_score = self.analyze_earnings_growth()
        ai_momentum_score = self.analyze_ai_momentum()

        # Calculate overall market sentiment
        overall_score = self.calculate_overall_market_sentiment(
            fed_policy_score, inflation_score, geopolitical_risk_score,
            earnings_growth_score, ai_momentum_score
        )

        return {
            'fed_policy_score': fed_policy_score,
            'inflation_score': inflation_score,
            'geopolitical_risk_score': geopolitical_risk_score,
            'earnings_growth_score': earnings_growth_score,
            'ai_momentum_score': ai_momentum_score,
            'overall_score': overall_score
        }

    def analyze_fed_policy(self):
        # Analyze Federal Reserve policy impact
        policy = self.market_factors.get('Fed Policy', '')
        if 'held rates steady' in policy:
            return 5
        elif 'rate cuts' in policy:
            return 8
        elif 'rate hikes' in policy:
            return 2
        return 5

    def analyze_inflation(self):
        # Analyze inflation impact on market
        inflation = self.market_factors.get('Inflation', '')
        if 'CPI at 2.4%' in inflation:
            return 6
        elif 'CPI above 3%' in inflation:
            return 3
        return 5

    def analyze_geopolitical_risk(self):
        # Analyze geopolitical tensions impact
        tensions = self.market_factors.get('Geopolitical Tensions', '')
        if 'tensions in Middle East' in tensions:
            return 3
        return 5

    def analyze_earnings_growth(self):
        # Analyze earnings growth impact
        earnings = self.market_factors.get('Earnings Growth', '')
        if 'growth of 12.8%' in earnings:
            return 8
        return 5

    def analyze_ai_momentum(self):
        # Analyze AI sector momentum
        ai_momentum = self.market_factors.get('AI Momentum', '')
        if 'momentum continuing' in ai_momentum:
            return 7
        return 5

    def calculate_overall_market_sentiment(self, *scores):
        # Calculate short-term and long-term market sentiment
        short_term_score = sum(scores) / len(scores)
        long_term_score = (short_term_score + 5) / 2
        return {'short_term': short_term_score, 'long_term': long_term_score}

class StockAnalyzer:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        # Define sector mapping for each stock
        self.sector_data = {
            'MSFT': 'Technology',
            'NVDA': 'Technology',
            'AAPL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'META': 'Communication Services',
            'GOOGL': 'Communication Services',
            'TSLA': 'Consumer Discretionary',
            'PLTR': 'Technology'
        }
        # Define financial data for each stock
        self.financial_data = {
            'MSFT': {'revenue': 245.1, 'growth': 15, 'notes': 'strong cloud growth'},
            'NVDA': {'revenue': 47.5, 'growth': 200, 'notes': 'AI leadership'},
            'AAPL': {'revenue': 46.2, 'growth': 6, 'services_revenue': 25, 'services_growth': 12},
            'AMZN': {'revenue': 187.8, 'growth': 10, 'aws_revenue': 28.8, 'aws_growth': 19},
            'META': {'ad_revenue': 46.8, 'ad_growth': 21, 'reality_labs_loss': 5},
            'GOOGL': {'search_revenue': 54, 'search_growth': 13, 'cloud_revenue': 12, 'cloud_growth': 30},
            'TSLA': {'deliveries': 2, 'energy_storage': 'all-time high'},
            'PLTR': {'growth': 36, 'us_growth': 52}
        }

    def analyze_stocks(self):
        # Perform comprehensive stock analysis
        analysis_results = {}
        for stock, price in self.stock_data.items():
            financial_metrics = self.financial_metrics_analysis(stock)
            technical_metrics = self.technical_analysis(stock)
            sector = self.sector_analysis(stock)
            risk_assessment = self.risk_assessment(stock)
            growth_potential = self.growth_potential(stock)
            valuation_metrics = self.valuation_metrics(stock)

            analysis_results[stock] = {
                'financial_metrics': financial_metrics,
                'technical_metrics': technical_metrics,
                'sector': sector,
                'risk_assessment': risk_assessment,
                'growth_potential': growth_potential,
                'valuation_metrics': valuation_metrics
            }
        return analysis_results

    def financial_metrics_analysis(self, stock):
        # Get financial metrics for specific stock
        return self.financial_data.get(stock, {})

    def technical_analysis(self, stock):
        # Generate technical analysis indicators
        return {
            'moving_average': np.random.uniform(0.95, 1.05),
            'rsi': np.random.uniform(30, 70)
        }

    def sector_analysis(self, stock):
        # Get sector information for stock
        return self.sector_data.get(stock, 'Unknown')

    def risk_assessment(self, stock):
        # Calculate risk metrics for stock
        return {
            'volatility': np.random.uniform(0.1, 0.3),
            'beta': np.random.uniform(0.8, 1.2)
        }

    def growth_potential(self, stock):
        # Assess growth potential of stock
        return {'growth_score': np.random.uniform(1, 10)}

    def valuation_metrics(self, stock):
        # Calculate valuation metrics
        return {
            'pe_ratio': np.random.uniform(10, 30),
            'pb_ratio': np.random.uniform(1, 5)
        }

class RecommendationEngine:
    def __init__(self, market_analyzer, stock_analyzer):
        self.market_analyzer = market_analyzer
        self.stock_analyzer = stock_analyzer

    def generate_recommendations(self):
        # Generate investment recommendations based on analysis
        market_sentiment = self.market_analyzer.analyze_factors()['overall_score']
        stock_analysis = self.stock_analyzer.analyze_stocks()
        recommendations = {}

        for stock, analysis in stock_analysis.items():
            financial_metrics = analysis['financial_metrics']
            technical_metrics = analysis['technical_metrics']
            sector = analysis['sector']
            risk_assessment = analysis['risk_assessment']
            growth_potential = analysis['growth_potential']
            valuation_metrics = analysis['valuation_metrics']

            recommendation, confidence, reasoning = self.evaluate_recommendation(
                stock, financial_metrics, technical_metrics, market_sentiment,
                risk_assessment, growth_potential, valuation_metrics, sector
            )

            recommendations[stock] = {
                'recommendation': recommendation,
                'confidence': confidence,
                'reasoning': reasoning,
                'price_target': self.calculate_price_target(self.stock_analyzer.stock_data[stock], growth_potential),
                'risk_level': risk_assessment['volatility'],
                'diversification': self.evaluate_diversification(stock, sector)
            }
        return recommendations

    def evaluate_recommendation(self, stock, financial_metrics, technical_metrics,
                                market_sentiment, risk_assessment, growth_potential,
                                valuation_metrics, sector):
        # Evaluate and generate recommendation for specific stock
        if financial_metrics.get('growth', 0) > 20 or 'AI leadership' in financial_metrics.get('notes', ''):
            return "Buy", "High", "Strong growth and leadership in key sectors."
        if risk_assessment['volatility'] > 0.25:
            return "Sell", "High", "High volatility and market uncertainty."
        return "Hold", "Medium", "Balanced financial and technical indicators."

    def calculate_price_target(self, current_price, growth_potential):
        # Calculate price target based on growth potential
        return current_price * (1 + growth_potential['growth_score'] / 100)

    def evaluate_diversification(self, stock, sector):
        # Evaluate diversification impact
        return f"Considered within {sector} sector."

class ReportGenerator:
    def __init__(self, market_analyzer, stock_analyzer, recommendation_engine):
        self.market_analyzer = market_analyzer
        self.stock_analyzer = stock_analyzer
        self.recommendation_engine = recommendation_engine
        self.output_dir = Path('reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_executive_summary(self):
        # Generate executive summary section
        market_sentiment = self.market_analyzer.analyze_factors()['overall_score']
        return f"Executive Summary:\nMarket Outlook: Short-term sentiment is {market_sentiment['short_term']:.2f}, Long-term sentiment is {market_sentiment['long_term']:.2f}.\n"

    def generate_market_factor_analysis(self):
        # Generate market factor analysis section
        analysis = self.market_analyzer.analyze_factors()
        return f"Market Factor Analysis:\nFed Policy Score: {analysis['fed_policy_score']}\nInflation Score: {analysis['inflation_score']}\nGeopolitical Risk Score: {analysis['geopolitical_risk_score']}\nEarnings Growth Score: {analysis['earnings_growth_score']}\nAI Momentum Score: {analysis['ai_momentum_score']}\n"

    def generate_stock_analysis_and_recommendations(self):
        # Generate stock analysis and recommendations section
        recommendations = self.recommendation_engine.generate_recommendations()
        analysis = "Stock Analysis and Recommendations:\n"
        for stock, details in recommendations.items():
            analysis += f"\n{stock}:\n  Recommendation: {details['recommendation']}\n  Confidence: {details['confidence']}\n  Reasoning: {details['reasoning']}\n  Price Target: {details['price_target']:.2f}\n  Risk Level: {details['risk_level']:.2f}\n  Diversification: {details['diversification']}\n"
        return analysis

    def generate_risk_assessment_and_portfolio_considerations(self):
        # Generate risk assessment section
        return "Risk Assessment and Portfolio Considerations:\n  Consider diversifying across sectors to mitigate risks.\n"

    def generate_key_metrics_and_performance_indicators(self):
        # Generate key metrics section
        return "Key Metrics and Performance Indicators:\n  Monitor P/E ratios and earnings growth for valuation insights.\n"

    def compile_report(self):
        # Compile all sections into comprehensive report
        return self.generate_executive_summary() + self.generate_market_factor_analysis() + self.generate_stock_analysis_and_recommendations() + self.generate_risk_assessment_and_portfolio_considerations() + self.generate_key_metrics_and_performance_indicators()

    def save_report(self, report):
        # Save report as text file
        text_file_path = self.output_dir / 'daily_report.txt'
        with open(text_file_path, 'w') as f:
            f.write(report)
        print(f"Report saved as text at {text_file_path}")

        # Save report as JSON file
        json_file_path = self.output_dir / 'daily_report.json'
        report_data = {
            "executive_summary": self.generate_executive_summary(),
            "market_factor_analysis": self.generate_market_factor_analysis(),
            "stock_analysis_and_recommendations": self.generate_stock_analysis_and_recommendations(),
            "risk_assessment_and_portfolio_considerations": self.generate_risk_assessment_and_portfolio_considerations(),
            "key_metrics_and_performance_indicators": self.generate_key_metrics_and_performance_indicators()
        }
        with open(json_file_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"Report saved as JSON at {json_file_path}")

class VisualizationManager:
    def __init__(self, stock_analyzer, market_analyzer, recommendation_engine):
        self.stock_analyzer = stock_analyzer
        self.market_analyzer = market_analyzer
        self.recommendation_engine = recommendation_engine
        self.output_dir = Path('visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set up matplotlib parameters for better visualization
        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'NanumGothic']
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300

    def generate_stock_price_performance_chart(self):
        # Generate stock price performance bar chart
        stock_data = self.stock_analyzer.stock_data
        stocks = list(stock_data.keys())
        prices = list(stock_data.values())
        plt.figure(figsize=(10, 6))
        plt.bar(stocks, prices, color='skyblue')
        plt.title('Stock Price Performance')
        plt.xlabel('Stocks')
        plt.ylabel('Current Price ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stock_price_performance.png')
        plt.close()

    def generate_market_sentiment_dashboard(self):
        # Generate market sentiment dashboard
        scores = self.market_analyzer.analyze_factors()
        labels = ['Fed Policy', 'Inflation', 'Geopolitical Risk', 'Earnings Growth', 'AI Momentum']
        values = [scores['fed_policy_score'], scores['inflation_score'], scores['geopolitical_risk_score'], scores['earnings_growth_score'], scores['ai_momentum_score']]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='lightgreen')
        plt.title('Market Sentiment Dashboard')
        plt.xlabel('Factors')
        plt.ylabel('Scores')
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'market_sentiment_dashboard.png')
        plt.close()

    def generate_risk_return_scatter_plot(self):
        # Generate risk-return scatter plot
        analysis_results = self.stock_analyzer.analyze_stocks()
        stocks = list(analysis_results.keys())
        risks = [analysis_results[stock]['risk_assessment']['volatility'] for stock in stocks]
        returns = [analysis_results[stock]['financial_metrics'].get('growth', 0) for stock in stocks]
        plt.figure(figsize=(10, 6))
        plt.scatter(risks, returns, color='orange')
        for i, stock in enumerate(stocks):
            plt.annotate(stock, (risks[i], returns[i]))
        plt.title('Risk-Return Scatter Plot')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Earnings Growth (Return)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_return_scatter_plot.png')
        plt.close()

    def generate_sector_distribution_chart(self):
        # Generate sector distribution pie chart
        sectors = list(self.stock_analyzer.sector_data.values())
        sector_counts = pd.Series(sectors).value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Sector Distribution')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sector_distribution_chart.png')
        plt.close()

    def generate_recommendation_summary_pie_chart(self):
        # Generate recommendation summary pie chart
        recommendations = self.recommendation_engine.generate_recommendations()
        recommendation_counts = pd.Series([rec['recommendation'] for rec in recommendations.values()]).value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(recommendation_counts, labels=recommendation_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Recommendation Summary')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recommendation_summary_pie_chart.png')
        plt.close()

    def generate_technical_indicators_visualization(self):
        # Generate technical indicators visualization
        analysis_results = self.stock_analyzer.analyze_stocks()
        stocks = list(analysis_results.keys())
        moving_averages = [analysis_results[stock]['technical_metrics']['moving_average'] for stock in stocks]
        rsis = [analysis_results[stock]['technical_metrics']['rsi'] for stock in stocks]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Stocks')
        ax1.set_ylabel('Moving Average Multiplier', color='tab:blue')
        ax1.bar(stocks, moving_averages, color='tab:blue', alpha=0.6, label='Moving Average')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('RSI', color='tab:red')
        ax2.plot(stocks, rsis, color='tab:red', marker='o', label='RSI')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        fig.tight_layout()
        plt.title('Technical Indicators Visualization')
        plt.xticks(rotation=45)
        plt.savefig(self.output_dir / 'technical_indicators_visualization.png')
        plt.close()

class DailyStockAnalysisAgent:
    def __init__(self, market_factors, stock_data):
        self.market_factors = market_factors
        self.stock_data = stock_data
        # Initialize all analysis components
        self.market_analyzer = MarketFactorAnalyzer(market_factors)
        self.stock_analyzer = StockAnalyzer(stock_data)
        self.recommendation_engine = RecommendationEngine(self.market_analyzer, self.stock_analyzer)
        self.report_generator = ReportGenerator(self.market_analyzer, self.stock_analyzer, self.recommendation_engine)
        self.visualization_manager = VisualizationManager(self.stock_analyzer, self.market_analyzer, self.recommendation_engine)
        self.setup_logging()

    def setup_logging(self):
        # Configure logging for the application
        logging.basicConfig(filename='analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("DailyStockAnalysisAgent initialized.")

    def run_daily_analysis(self):
        # Execute complete daily analysis workflow
        logging.info("Starting daily stock market analysis.")
        market_analysis = self.market_analyzer.analyze_factors()
        logging.info(f"Market analysis completed: {market_analysis}")
        stock_analysis = self.stock_analyzer.analyze_stocks()
        logging.info(f"Stock analysis completed: {stock_analysis}")
        recommendations = self.recommendation_engine.generate_recommendations()
        logging.info(f"Recommendations generated: {recommendations}")
        comprehensive_report = self.report_generator.compile_report()
        self.report_generator.save_report(comprehensive_report)
        logging.info("Report generated and saved.")
        # Generate all visualizations
        self.visualization_manager.generate_stock_price_performance_chart()
        self.visualization_manager.generate_market_sentiment_dashboard()
        self.visualization_manager.generate_risk_return_scatter_plot()
        self.visualization_manager.generate_sector_distribution_chart()
        self.visualization_manager.generate_recommendation_summary_pie_chart()
        self.visualization_manager.generate_technical_indicators_visualization()
        logging.info("Visualizations generated.")
        summary = self.generate_summary(market_analysis, stock_analysis, recommendations)
        print(summary)
        logging.info("Daily analysis completed successfully.")

    def generate_summary(self, market_analysis, stock_analysis, recommendations):
        # Generate summary of analysis results
        summary = "Daily Stock Market Analysis Summary:\n"
        summary += f"Market Sentiment: {market_analysis['overall_score']}\n"
        summary += "Stock Recommendations:\n"
        for stock, details in recommendations.items():
            summary += f"  {stock}: {details['recommendation']} with confidence {details['confidence']}\n"
        return summary

def send_email_with_report(report_generator, visualization_manager):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    from email.mime.base import MIMEBase
    from email import encoders

    # Get email credentials from environment variables
    gmail_user = os.environ.get('GMAIL_USERNAME')
    gmail_password = os.environ.get('GMAIL_PASSWORD')
    recipient_email = os.environ.get('RECIPIENT_EMAIL')

    # Check if all required environment variables are set
    if not all([gmail_user, gmail_password, recipient_email]):
        print("⚠️ 이메일 관련 환경 변수가 설정되지 않아 이메일을 발송할 수 없습니다.")
        return

    # Create email message
    msg = MIMEMultipart()
    current_date = datetime.now().strftime('%Y-%m-%d')
    msg['Subject'] = f"📊 일일 주식 분석 리포트 - {current_date}"
    msg['From'] = gmail_user
    msg['To'] = recipient_email
    
    # Generate email body with HTML formatting
    summary = report_generator.generate_executive_summary()
    html_body = f'''<html>
    <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #2E86AB;">📊 일일 주식 분석 리포트</h2>
        <h3 style="color: #333;">📅 {current_date}</h3>
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; font-size: 14px; line-height: 1.6;">
            <pre>{summary}</pre>
        </div>
        <p>상세 내용은 첨부된 리포트 파일과 차트 이미지를 확인해주세요.</p>
        <hr>
        <p style="color: #6c757d; font-size: 12px;">
            <em>🤖 이 리포트는 GitHub Actions를 통해 자동으로 생성되었습니다.</em>
        </p>
    </body>
    </html>'''
    
    # Attach HTML body to email - FIXED: Changed MimeText to MIMEText
    msg.attach(MIMEText(html_body, 'html'))

    # Define list of files to attach
    attachments = [
        report_generator.output_dir / 'daily_report.txt',
        visualization_manager.output_dir / 'stock_price_performance.png',
        visualization_manager.output_dir / 'market_sentiment_dashboard.png',
        visualization_manager.output_dir / 'risk_return_scatter_plot.png',
        visualization_manager.output_dir / 'sector_distribution_chart.png',
    ]
    
    # Process each attachment
    for file_path in attachments:
        try:
            with open(file_path, 'rb') as attachment:
                if file_path.suffix == '.png':
                    # Handle image files
                    part = MIMEImage(attachment.read(), name=os.path.basename(file_path))
                else:
                    # Handle other file types
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                
                # Add attachment header
                part.add_header(
                    'Content-Disposition', 
                    f'attachment; filename="{os.path.basename(file_path)}"'
                )
                msg.attach(part)
        except FileNotFoundError:
            print(f"경고: 첨부 파일을 찾을 수 없습니다 - {file_path}")
    
    # Send email via SMTP
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Enable encryption
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        server.quit()
        print('✅ 이메일이 성공적으로 발송되었습니다!')
    except Exception as e:
        print(f'❌ 이메일 발송 중 오류 발생: {e}')
        raise

if __name__ == "__main__":
    try:
        # Load configuration from JSON file
        config_path = 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        market_factors = config['market_factors']
        stock_data = config['stock_data']
        
        # Initialize and run analysis agent
        analysis_agent = DailyStockAnalysisAgent(market_factors, stock_data)
        
        # Execute daily analysis workflow
        analysis_agent.run_daily_analysis()
        # Send email with results
        send_email_with_report(analysis_agent.report_generator, analysis_agent.visualization_manager)
    except Exception as e:
        logging.error(f"스크립트 실행 중 치명적인 오류 발생: {e}", exc_info=True)
        print(f"스크립트 실행 중 치명적인 오류 발생: {e}")
        exit(1)
