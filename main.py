import os
import sys
import logging
from typing import List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analyzer import YahooFinanceNewsAnalyzer
from src.utils import setup_logging, send_webhook_notification, estimate_api_cost
from config.settings import settings

def main():
    """
    Main function to run the complete Yahoo Finance + ChatGPT analysis system
    """
    # Setup logging
    setup_logging()
    
    # Validate settings
    if not settings.validate():
        return
    
    try:
        print("🚀 Starting Enhanced Stock Analysis Pipeline...")
        print(f"📊 Analyzing {len(settings.DEFAULT_TICKERS)} stocks: {', '.join(settings.DEFAULT_TICKERS)}")
        
        # Show cost estimation
        cost_estimate = estimate_api_cost(len(settings.DEFAULT_TICKERS))
        print(f"💰 Estimated API cost: ${cost_estimate['estimated_cost']:.2f}")
        
        # Ask for confirmation if cost is high
        if cost_estimate['estimated_cost'] > 5.0:  # More than $5
            proceed = input(f"\nEstimated cost is ${cost_estimate['estimated_cost']:.2f}. Proceed? (y/n): ").lower()
            if proceed != 'y':
                print("Analysis cancelled.")
                return
        
        # Initialize analyzer
        analyzer = YahooFinanceNewsAnalyzer()
        
        # Run complete analysis
        print("🔍 Collecting data and running AI analysis...")
        results = analyzer.run_complete_analysis(settings.DEFAULT_TICKERS)
        
        # Send webhook notification if configured
        if settings.WEBHOOK_URL:
            print("📢 Sending analysis notification...")
            summary_message = create_summary_message(results)
            send_webhook_notification(settings.WEBHOOK_URL, summary_message)
        
        # Print comprehensive summary
        print_analysis_summary(results)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print("Check the stock_analysis.log file for more details.")

def create_summary_message(results: Dict) -> str:
    """Create summary message for webhook notification"""
    try:
        successful = results['summary']['successful_analyses']
        total = results['summary']['total_stocks']
        
        message = f"""
📊 **Stock Analysis Complete** 📊

**Date:** {results['analysis_timestamp'][:10]}
**Success Rate:** {successful}/{total} stocks analyzed

**Top Recommendations:**
"""
        
        # Add top 3 recommendations
        for analysis in results['individual_analyses'][:3]:
            if 'error' not in analysis:
                rec = analysis['structured_analysis'].get('investment_recommendation', 'N/A')
                rec_summary = rec.split('.')[0] if '.' in rec else rec[:50]
                message += f"• {analysis['ticker']}: {rec_summary}\n"
        
        if results['cost_estimate']:
            message += f"\n💰 **API Cost:** ${results['cost_estimate']['estimated_cost']:.2f}"
        
        return message
        
    except Exception as e:
        return f"Analysis completed with {results['summary']['successful_analyses']} stocks analyzed."

def print_analysis_summary(results: Dict) -> None:
    """Print detailed analysis summary to console"""
    print(f"\n{'='*80}")
    print("📋 ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print(f"📅 Analysis Date: {results['analysis_timestamp'][:19]}")
    print(f"📊 Stocks Analyzed: {results['summary']['total_stocks']}")
    print(f"✅ Successful Analyses: {results['summary']['successful_analyses']}")
    print(f"❌ Failed Analyses: {results['summary']['failed_analyses']}")
    print(f"💰 Estimated Cost: ${results['cost_estimate']['estimated_cost']:.2f}")
    
    print(f"\n🎯 TOP RECOMMENDATIONS:")
    for analysis in results['individual_analyses'][:5]:  # Top 5
        if 'error' not in analysis:
            rec = analysis['structured_analysis'].get('investment_recommendation', 'N/A')
            rec_summary = rec.split('.')[0] if '.' in rec else rec[:100]
            print(f"  • {analysis['ticker']}: {rec_summary}...")
    
    print(f"\n💾 SAVED FILES:")
    print(f"  • Analysis data: {settings.ANALYSIS_DIR}/ directory")
    print(f"  • Technical charts: {settings.CHARTS_DIR}/ directory")
    
    # Calculate actual tokens used
    total_tokens = sum([
        analysis.get('tokens_used', 0) 
        for analysis in results['individual_analyses']
        if 'tokens_used' in analysis
    ])
    if 'portfolio_analysis' in results and 'tokens_used' in results['portfolio_analysis']:
        total_tokens += results['portfolio_analysis']['tokens_used']
    
    if total_tokens > 0:
        actual_cost = (total_tokens / 1000) * 0.015  # Rough average pricing
        print(f"📊 Actual tokens used: {total_tokens:,}")
        print(f"💰 Actual estimated cost: ~${actual_cost:.2f}")
    
    print(f"\n🎉 Analysis pipeline completed successfully!")

if __name__ == "__main__":
    main()
