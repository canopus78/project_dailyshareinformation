
import logging
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime

class MarketFactorAnalyzer:
    def __init__(self, market_factors):
        self.market_factors = market_factors

    def analyze_factors(self):
        fed_policy_score = self.analyze_fed_policy()
        inflation_score = self.analyze_inflation()
        geopolitical_risk_score = self.analyze_geopolitical_risk()
        earnings_growth_score = self.analyze_earnings_growth()
        ai_momentum_score = self.analyze_ai_momentum()

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
        policy = self.market_factors.get('Fed Policy', '')
        if 'held rates steady' in policy:
            return 5
        elif 'rate cuts' in policy:
            return 8
        elif 'rate hikes' in policy:
            return 2
        return 5

    def analyze_inflation(self):
        inflation = self.market_factors.get('Inflation', '')
        if 'CPI at 2.4%' in inflation:
            return 6
        elif 'CPI above 3%' in inflation:
            return 3
        return 5

    def analyze_geopolitical_risk(self):
        tensions = self.market_factors.get('Geopolitical Tensions', '')
        if 'tensions in Middle East' in tensions:
            return 3
        return 5

    def analyze_earnings_growth(self):
        earnings = self.market_factors.get('Earnings Growth', '')
        if 'growth of 12.8%' in earnings:
            return 8
        return 5

    def analyze_ai_momentum(self):
        ai_momentum = self.market_factors.get('AI Momentum', '')
        if 'momentum continuing' in ai_momentum:
            return 7
        return 5

    def calculate_overall_market_sentiment(self, *scores):
        short_term_score = sum(scores) / len(scores)
        long_term_score = (short_term_score + 5) / 2
        return {'short_term': short_term_score, 'long_term': long_term_score}

class StockAnalyzer:
    def __init__(self, stock_data):
        self.stock_data = stock_data
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
        data = self.financial_data.get(stock, {})
        return data

    def technical_analysis(self, stock):
        moving_average = np.random.uniform(0.95, 1.05)
        rsi = np.random.uniform(30, 70)
        return {
            'moving_average': moving_average,
            'rsi': rsi
        }

    def sector_analysis(self, stock):
        return self.sector_data.get(stock, 'Unknown')

    def risk_assessment(self, stock):
        volatility = np.random.uniform(0.1, 0.3)
        beta = np.random.uniform(0.8, 1.2)
        return {
            'volatility': volatility,
            'beta': beta
        }

    def growth_potential(self, stock):
        growth_score = np.random.uniform(1, 10)
        return {
            'growth_score': growth_score
        }

    def valuation_metrics(self, stock):
        pe_ratio = np.random.uniform(10, 30)
        pb_ratio = np.random.uniform(1, 5)
        return {
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio
        }

class RecommendationEngine:
    def __init__(self, market_analyzer, stock_analyzer):
        self.market_analyzer = market_analyzer
        self.stock_analyzer = stock_analyzer

    def generate_recommendations(self):
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
        recommendation = "Hold"
        confidence = "Medium"
        reasoning = "Balanced financial and technical indicators."

        # Enhanced logic based on real financial data
        if financial_metrics.get('growth', 0) > 20 or 'AI leadership' in financial_metrics.get('notes', ''):
            recommendation = "Buy"
            confidence = "High"
            reasoning = "Strong growth and leadership in key sectors."
        elif risk_assessment['volatility'] > 0.25:
            recommendation = "Sell"
            confidence = "High"
            reasoning = "High volatility and market uncertainty."

        return recommendation, confidence, reasoning

    def calculate_price_target(self, current_price, growth_potential):
        return current_price * (1 + growth_potential['growth_score'] / 100)

    def evaluate_diversification(self, stock, sector):
        return f"Considered within {sector} sector."

class ReportGenerator:
    def __init__(self, market_analyzer, stock_analyzer, recommendation_engine):
        self.market_analyzer = market_analyzer
        self.stock_analyzer = stock_analyzer
        self.recommendation_engine = recommendation_engine
        self.output_dir = Path('/home/user/stock_analysis/reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_executive_summary(self):
        market_sentiment = self.market_analyzer.analyze_factors()['overall_score']
        summary = (
            f"Executive Summary:\n"
            f"Market Outlook: Short-term sentiment is {market_sentiment['short_term']:.2f}, "
            f"Long-term sentiment is {market_sentiment['long_term']:.2f}.\n"
        )
        return summary

    def generate_market_factor_analysis(self):
        analysis = self.market_analyzer.analyze_factors()
        return (
            f"Market Factor Analysis:\n"
            f"Fed Policy Score: {analysis['fed_policy_score']}\n"
            f"Inflation Score: {analysis['inflation_score']}\n"
            f"Geopolitical Risk Score: {analysis['geopolitical_risk_score']}\n"
            f"Earnings Growth Score: {analysis['earnings_growth_score']}\n"
            f"AI Momentum Score: {analysis['ai_momentum_score']}\n"
        )

    def generate_stock_analysis_and_recommendations(self):
        recommendations = self.recommendation_engine.generate_recommendations()
        analysis = "Stock Analysis and Recommendations:\n"
        for stock, details in recommendations.items():
            analysis += (
                f"\n{stock}:\n"
                f"  Recommendation: {details['recommendation']}\n"
                f"  Confidence: {details['confidence']}\n"
                f"  Reasoning: {details['reasoning']}\n"
                f"  Price Target: {details['price_target']:.2f}\n"
                f"  Risk Level: {details['risk_level']:.2f}\n"
                f"  Diversification: {details['diversification']}\n"
            )
        return analysis

    def generate_risk_assessment_and_portfolio_considerations(self):
        risk_assessment = "Risk Assessment and Portfolio Considerations:\n"
        risk_assessment += "  Consider diversifying across sectors to mitigate risks.\n"
        return risk_assessment

    def generate_key_metrics_and_performance_indicators(self):
        metrics = "Key Metrics and Performance Indicators:\n"
        metrics += "  Monitor P/E ratios and earnings growth for valuation insights.\n"
        return metrics

    def compile_report(self):
        report = (
            self.generate_executive_summary() +
            self.generate_market_factor_analysis() +
            self.generate_stock_analysis_and_recommendations() +
            self.generate_risk_assessment_and_portfolio_considerations() +
            self.generate_key_metrics_and_performance_indicators()
        )
        return report

    def save_report(self, report):
        text_file_path = self.output_dir / 'daily_report.txt'
        with open(text_file_path, 'w') as f:
            f.write(report)
        print(f"Report saved as text at {text_file_path}")

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
        self.output_dir = Path('/home/user/stock_analysis/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['font.family'] = ['Noto Sans CJK JP']
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300

    def generate_stock_price_performance_chart(self):
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
        plt.show()

    def generate_market_sentiment_dashboard(self):
        scores = self.market_analyzer.analyze_factors()
        labels = ['Fed Policy', 'Inflation', 'Geopolitical Risk', 'Earnings Growth', 'AI Momentum']
        values = [scores['fed_policy_score'], scores['inflation_score'], scores['geopolitical_risk_score'],
                  scores['earnings_growth_score'], scores['ai_momentum_score']]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='lightgreen')
        plt.title('Market Sentiment Dashboard')
        plt.xlabel('Factors')
        plt.ylabel('Scores')
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'market_sentiment_dashboard.png')
        plt.show()

    def generate_risk_return_scatter_plot(self):
        analysis_results = self.stock_analyzer.analyze_stocks()
        stocks = list(analysis_results.keys())
        risks = [analysis_results[stock]['risk_assessment']['volatility'] for stock in stocks]
        returns = [analysis_results[stock]['financial_metrics']['earnings_growth'] for stock in stocks]

        plt.figure(figsize=(10, 6))
        plt.scatter(risks, returns, color='orange')
        for i, stock in enumerate(stocks):
            plt.annotate(stock, (risks[i], returns[i]))
        plt.title('Risk-Return Scatter Plot')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Earnings Growth (Return)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_return_scatter_plot.png')
        plt.show()

    def generate_sector_distribution_chart(self):
        sector_data = self.stock_analyzer.sector_data
        sectors = list(sector_data.values())
        sector_counts = pd.Series(sectors).value_counts()

        plt.figure(figsize=(8, 6))
        plt.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Sector Distribution')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sector_distribution_chart.png')
        plt.show()

    def generate_recommendation_summary_pie_chart(self):
        recommendations = self.recommendation_engine.generate_recommendations()
        recommendation_counts = pd.Series([rec['recommendation'] for rec in recommendations.values()]).value_counts()

        plt.figure(figsize=(8, 6))
        plt.pie(recommendation_counts, labels=recommendation_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Recommendation Summary')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recommendation_summary_pie_chart.png')
        plt.show()

    def generate_technical_indicators_visualization(self):
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
        plt.show()

class DailyStockAnalysisAgent:
    def __init__(self, market_factors, stock_data):
        self.market_factors = market_factors
        self.stock_data = stock_data
        self.market_analyzer = MarketFactorAnalyzer(market_factors)
        self.stock_analyzer = StockAnalyzer(stock_data)
        self.recommendation_engine = RecommendationEngine(self.market_analyzer, self.stock_analyzer)
        self.report_generator = ReportGenerator(self.market_analyzer, self.stock_analyzer, self.recommendation_engine)
        self.visualization_manager = VisualizationManager(self.stock_analyzer, self.market_analyzer, self.recommendation_engine)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename='/home/user/stock_analysis/analysis.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("DailyStockAnalysisAgent initialized.")

    def update_market_factors(self, new_market_factors):
        self.market_factors.update(new_market_factors)
        self.market_analyzer = MarketFactorAnalyzer(self.market_factors)
        logging.info("Market factors updated.")

    def update_stock_data(self, new_stock_data):
        self.stock_data.update(new_stock_data)
        self.stock_analyzer = StockAnalyzer(self.stock_data)
        logging.info("Stock data updated.")

    def configure_analysis(self, **kwargs):
        logging.info(f"Analysis configuration updated with parameters: {kwargs}")

    def run_daily_analysis(self):
        try:
            logging.info("Starting daily stock market analysis.")

            # Analyze market factors
            market_analysis = self.market_analyzer.analyze_factors()
            logging.info(f"Market analysis completed: {market_analysis}")

            # Analyze stocks
            stock_analysis = self.stock_analyzer.analyze_stocks()
            logging.info(f"Stock analysis completed: {stock_analysis}")

            # Generate recommendations
            recommendations = self.recommendation_engine.generate_recommendations()
            logging.info(f"Recommendations generated: {recommendations}")

            # Generate report
            comprehensive_report = self.report_generator.compile_report()
            self.report_generator.save_report(comprehensive_report)
            logging.info("Report generated and saved.")

            # Generate visualizations
            self.visualization_manager.generate_stock_price_performance_chart()
            self.visualization_manager.generate_market_sentiment_dashboard()
            self.visualization_manager.generate_risk_return_scatter_plot()
            self.visualization_manager.generate_sector_distribution_chart()
            self.visualization_manager.generate_recommendation_summary_pie_chart()
            self.visualization_manager.generate_technical_indicators_visualization()
            logging.info("Visualizations generated.")

            # Output summary
            summary = self.generate_summary(market_analysis, stock_analysis, recommendations)
            print(summary)
            logging.info("Daily analysis completed successfully.")

        except Exception as e:
            logging.error(f"Error during daily analysis: {e}")
            print(f"An error occurred during the analysis: {e}")

    def generate_summary(self, market_analysis, stock_analysis, recommendations):
        summary = "Daily Stock Market Analysis Summary:\n"
        summary += f"Market Sentiment: {market_analysis['overall_score']}\n"
        summary += "Stock Recommendations:\n"
        for stock, details in recommendations.items():
            summary += f"  {stock}: {details['recommendation']} with confidence {details['confidence']}\n"
        return summary




def send_email_with_report(report_generator, visualization_manager):
    """
    분석 리포트와 시각화 자료를 이메일로 전송하는 함수
    """
    gmail_user = os.environ.get('GMAIL_USERNAME')
    gmail_password = os.environ.get('GMAIL_PASSWORD')
    recipient_email = os.environ.get('RECIPIENT_EMAIL')

    if not all([gmail_user, gmail_password, recipient_email]):
        print("⚠️ 이메일 관련 환경 변수가 설정되지 않아 이메일을 발송할 수 없습니다.")
        return

    # 이메일 메시지 생성
    msg = MIMEMultipart()
    current_date = datetime.now().strftime('%Y-%m-%d')
    msg['Subject'] = f"📊 일일 주식 분석 리포트 - {current_date}"
    msg['From'] = gmail_user
    msg['To'] = recipient_email

    # HTML 본문 생성 (요약 정보 포함)
    summary = report_generator.generate_executive_summary()
    html_body = f"""
    <html>
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
    </html>
    """
    msg.attach(MimeText(html_body, 'html'))

    # 첨부 파일 추가 (리포트, 차트)
    attachments = [
        report_generator.output_dir / 'daily_report.txt',
        visualization_manager.output_dir / 'stock_price_performance.png',
        visualization_manager.output_dir / 'market_sentiment_dashboard.png',
        visualization_manager.output_dir / 'risk_return_scatter_plot.png',
        visualization_manager.output_dir / 'sector_distribution_chart.png',
    ]

    for file_path in attachments:
        try:
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(file_path)}')
            msg.attach(part)
        except FileNotFoundError:
            print(f"경고: 첨부 파일을 찾을 수 없습니다 - {file_path}")


    # Gmail SMTP 서버를 통해 이메일 발송
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        server.quit()
        print('✅ 이메일이 성공적으로 발송되었습니다!')
    except Exception as e:
        print(f'❌ 이메일 발송 중 오류 발생: {e}')


if __name__ == "__main__":
    # 설정 파일 로드
    config_path = 'config' # 파일 이름을 'config'로 가정
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        market_factors = config['market_factors']
        stock_data = config['stock_data']

        # 에이전트 실행
        analysis_agent = DailyStockAnalysisAgent(market_factors, stock_data)
        analysis_agent.run_daily_analysis()

        # 이메일 발송
        send_email_with_report(analysis_agent.report_generator, analysis_agent.visualization_manager)

    except FileNotFoundError:
        print(f"오류: 설정 파일 '{config_path}'을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print(f"오류: 설정 파일 '{config_path}'의 형식이 올바르지 않습니다.")
    except Exception as e:
        logging.error(f"스크립트 실행 중 치명적인 오류 발생: {e}")
        print(f"스크립트 실행 중 치명적인 오류 발생: {e}")
