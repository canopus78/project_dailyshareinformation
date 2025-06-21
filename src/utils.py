import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import requests

def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('stock_analysis.log')
        ]
    )

def ensure_directories_exist(directories: List[str]) -> None:
    """
    Ensure that required directories exist
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create .gitkeep file if directory is empty
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.listdir(directory) and not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                f.write('')

def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {str(e)}")
        return False

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data or empty dict if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {str(e)}")
        return {}

def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Percentage change
    """
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100

def format_currency(amount: float) -> str:
    """
    Format number as currency
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    if amount >= 1e12:
        return f"${amount/1e12:.2f}T"
    elif amount >= 1e9:
        return f"${amount/1e9:.2f}B"
    elif amount >= 1e6:
        return f"${amount/1e6:.2f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.2f}K"
    else:
        return f"${amount:.2f}"

def send_webhook_notification(webhook_url: str, message: str, username: str = "Stock Analysis Bot") -> bool:
    """
    Send notification to webhook (Slack, Discord, etc.)
    
    Args:
        webhook_url: Webhook URL
        message: Message to send
        username: Bot username
        
    Returns:
        True if successful, False otherwise
    """
    try:
        payload = {
            "text": message,
            "username": username
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        
        logging.info("Webhook notification sent successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error sending webhook notification: {str(e)}")
        return False

def clean_text(text: str) -> str:
    """
    Clean and normalize text data
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Remove common HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' '
    }
    
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    return text

def get_timestamp() -> str:
    """
    Get current timestamp as string
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validate_ticker(ticker: str) -> bool:
    """
    Basic validation for stock ticker symbols
    
    Args:
        ticker: Stock ticker to validate
        
    Returns:
        True if ticker appears valid
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic checks: 1-5 characters, alphanumeric, uppercase
    ticker = ticker.strip().upper()
    return len(ticker) >= 1 and len(ticker) <= 5 and ticker.isalnum()

def estimate_api_cost(num_stocks: int, tokens_per_stock: int = 2000) -> Dict[str, float]:
    """
    Estimate OpenAI API costs for analysis
    
    Args:
        num_stocks: Number of stocks to analyze
        tokens_per_stock: Estimated tokens per stock
        
    Returns:
        Cost estimation details
    """
    portfolio_tokens = 1000  # Additional tokens for portfolio analysis
    total_tokens = (num_stocks * tokens_per_stock) + portfolio_tokens
    
    # GPT-4 Turbo pricing (approximate)
    input_cost_per_1k = 0.01   # $0.01 per 1K input tokens
    output_cost_per_1k = 0.03  # $0.03 per 1K output tokens
    
    # Assume 70% input, 30% output
    input_tokens = total_tokens * 0.7
    output_tokens = total_tokens * 0.3
    
    estimated_cost = (input_tokens / 1000 * input_cost_per_1k) + (output_tokens / 1000 * output_cost_per_1k)
    
    return {
        'total_tokens': total_tokens,
        'estimated_cost': estimated_cost,
        'cost_per_stock': estimated_cost / num_stocks if num_stocks > 0 else 0,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens
    }
