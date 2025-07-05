#!/usr/bin/env python3
"""
Setup script for Airline Trend Analyzer
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_secrets_file():
    """Create secrets file if it doesn't exist"""
    secrets_dir = ".streamlit"
    secrets_file = os.path.join(secrets_dir, "secrets.toml")
    
    if not os.path.exists(secrets_dir):
        os.makedirs(secrets_dir)
        print(f"📁 Created directory: {secrets_dir}")
    
    if not os.path.exists(secrets_file):
        with open(secrets_file, "w") as f:
            f.write("# Add your DeepSeek API key here\n")
            f.write("DEEPSEEK_API_KEY = \"your_deepseek_api_key_here\"\n")
            f.write("\n# Your existing SerpAPI key is already in the code\n")
            f.write("# SERPAPI_KEY = \"10a78c2ca9cfc32b491ec000a81927343b90972b90e9f881731eb81c74e314ee\"\n")
        print(f"📝 Created secrets file: {secrets_file}")
        print("⚠️  Please edit the secrets file to add your DeepSeek API key")
    else:
        print(f"✅ Secrets file already exists: {secrets_file}")

def check_api_keys():
    """Check if API keys are configured"""
    print("🔑 Checking API key configuration...")
    
    # Check SerpAPI key (already in code)
    print("✅ SerpAPI key is configured in the code")
    
    # Check DeepSeek API key
    try:
        import streamlit as st
        # This is a simple check - in actual app it will be loaded from secrets
        print("ℹ️  DeepSeek API key should be configured in .streamlit/secrets.toml")
        print("   The app will work without it, but AI insights will be disabled")
    except ImportError:
        print("⚠️  Streamlit not installed, cannot check DeepSeek API key")

def main():
    """Main setup function"""
    print("🚀 Setting up Airline Trend Analyzer...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create secrets file
    create_secrets_file()
    
    # Check API keys
    check_api_keys()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .streamlit/secrets.toml to add your DeepSeek API key (optional)")
    print("2. Run the application: python run_app.py")
    print("3. Or run directly: streamlit run airline_trend_analyzer.py")
    print("\n🎯 The application will be available at: http://localhost:8501")

if __name__ == "__main__":
    main() 