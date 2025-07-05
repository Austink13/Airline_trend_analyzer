#!/usr/bin/env python3
"""
Launcher script for Airline Trend Analyzer
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    print("✈️ Starting Airline Booking Market Trend Analyzer...")
    print("📊 Loading application...")
    
    try:
        # Check if required packages are installed
        import streamlit
        import requests
        import pandas
        import plotly
        import altair
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return
    
    # Launch the Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "airline_trend_analyzer.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main() 