#!/usr/bin/env python
import argparse, logging
from yahoo_finance_news_analyzer import YahooFinanceNewsAnalyzer

logging.basicConfig(level=logging.INFO)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="AAPL,GOOGL,MSFT 처럼 입력")
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    logging.info(f">>> Tickers: {tickers}")

    analyzer = YahooFinanceNewsAnalyzer()
    analyzer.run_complete_analysis(tickers)

if __name__ == "__main__":
    main()
