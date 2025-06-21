import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """
    Application settings and configuration
    """
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    WEBHOOK_URL: str = os.getenv('WEBHOOK_URL', '')
    
    # Default stock tickers to analyze
    DEFAULT_TICKERS: List[str] = [
        'AAPL',   # Apple Inc.
        'GOOGL',  # Alphabet Inc.
        'MSFT',   # Microsoft Corporation
        'TSLA',   # Tesla Inc.
        'NVDA',   # NVIDIA Corporation
        'AMZN',   # Amazon.com Inc.
        'META',   # Meta Platforms Inc.
        'NFLX'    # Netflix Inc.
    ]
    
    # Analysis settings
    MAX_STOCKS_PER_RUN: int = int(os.getenv('MAX_STOCKS_PER_RUN', '8'))
    HISTORICAL_PERIOD: str = '3mo'  # 3 months of historical data
    
    # OpenAI settings
    OPENAI_MODEL: str = 'gpt-4-turbo-preview'
    MAX_TOKENS: int = 2500
    TEMPERATURE: float = 0.3
    
    # File paths
    DATA_DIR: str = 'data'
    ANALYSIS_DIR: str = 'analysis'
    CHARTS_DIR: str = 'charts'
    
    # Rate limiting
    REQUEST_DELAY: float = 2.0  # seconds between requests
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required settings are present
        
        Returns:
            True if all required settings are valid
        """
        if not cls.OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY is required")
            print("Get your API key from: https://platform.openai.com/api-keys")
            return False
        
        return True

# Create global settings instance
settings = Settings()
