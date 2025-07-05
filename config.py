"""
Configuration file for Airline Trend Analyzer
"""

import os
import streamlit as st

# API Configuration
SERPAPI_KEY = "10a78c2ca9cfc32b491ec000a81927343b90972b90e9f881731eb81c74e314ee"
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")

# Application Settings
CURRENCY = "AUD"
DEFAULT_ANALYSIS_DAYS = 14
MAX_ANALYSIS_DAYS = 30
RATE_LIMIT_DELAY = 0.5  # seconds between API calls

# Popular Routes for Analysis
POPULAR_ROUTES = [
    ("Sydney", "Melbourne"),
    ("Melbourne", "Sydney"),
    ("Sydney", "Brisbane"),
    ("Brisbane", "Sydney"),
    ("Sydney", "Perth"),
    ("Perth", "Sydney"),
    ("Melbourne", "Brisbane"),
    ("Brisbane", "Melbourne"),
    ("Melbourne", "Perth"),
    ("Perth", "Melbourne"),
    ("Sydney", "Adelaide"),
    ("Adelaide", "Sydney"),
    ("Melbourne", "Adelaide"),
    ("Adelaide", "Melbourne"),
    ("Brisbane", "Perth"),
    ("Perth", "Brisbane")
]

# Extended IATA codes for Australian cities
IATA_CODES = {
    "Sydney": "SYD",
    "Melbourne": "MEL", 
    "Brisbane": "BNE",
    "Perth": "PER",
    "Adelaide": "ADL",
    "Gold Coast": "OOL",
    "Cairns": "CNS",
    "Darwin": "DRW",
    "Hobart": "HBA",
    "Canberra": "CBR",
    "Newcastle": "NTL",
    "Townsville": "TSV",
    "Alice Springs": "ASP",
    "Launceston": "LST",
    "Rockhampton": "ROK"
}

# Analysis Settings
PRICE_THRESHOLDS = {
    "low_demand": 0.8,  # 80% of mean price
    "high_demand": 1.2   # 120% of mean price
}

# Visualization Settings
CHART_HEIGHT = 400
CHART_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728"
}

# DeepSeek API Configuration
DEEPSEEK_CONFIG = {
    "model": "deepseek-chat",
    "temperature": 0.7,
    "max_tokens": 500,
    "timeout": 30,
    "enabled": True  # Set to False to disable DeepSeek API entirely
} 