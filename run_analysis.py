import argparse
import logging
from src.yahoo_finance_news_analyzer import YahooFinanceNewsAnalyzer   # ← 수정 완료

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated list, e.g. AAPL,GOOGL,MSFT"
    )
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    logging.info(f">>> Tickers: {tickers}")
    analyzer = YahooFinanceNewsAnalyzer()
    analyzer.run_complete_analysis(tickers)

if __name__ == "__main__":
    main()
