# ✈️ Airline Booking Market Trend Analyzer

A comprehensive web application for analyzing airline booking market trends, focusing on popular routes, price trends, high demand periods, and location-based insights using Google Flights API (via SerpAPI) and DeepSeek AI for intelligent insights.

## 🚀 Features

### 📊 **Single Route Analysis**
- **Price Trend Analysis**: Track price fluctuations over 7-30 days
- **Advanced Visualizations**: Interactive charts with Plotly
- **Demand Patterns**: Identify high/low demand periods
- **AI-Powered Insights**: DeepSeek integration for actionable recommendations

### 🏆 **Popular Routes Comparison**
- **Route Rankings**: Compare average prices across popular routes
- **Market Volatility**: Analyze price stability across routes
- **Cost Analysis**: Identify most affordable and expensive routes

### 📈 **Market Insights**
- **Comprehensive Analytics**: Market overview and trends
- **Seasonal Analysis**: Coming soon
- **Demand Forecasting**: AI-powered predictions (coming soon)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd airlines
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**:
   - Edit `.streamlit/secrets.toml` and add your DeepSeek API key
   - The SerpAPI key is already configured in the code

4. **Run the application**:
   ```bash
   streamlit run airline_trend_analyzer.py
   ```

## 🔧 Configuration

### API Keys Required:
- **SerpAPI Key**: Already configured (Google Flights API access)
- **DeepSeek API Key**: Add to `.streamlit/secrets.toml` for AI insights

### Supported Cities:
- Sydney (SYD)
- Melbourne (MEL)
- Brisbane (BNE)
- Perth (PER)
- Adelaide (ADL)
- Gold Coast (OOL)
- Cairns (CNS)
- Darwin (DRW)
- Hobart (HBA)
- Canberra (CBR)
- Newcastle (NTL)
- Townsville (TSV)
- Alice Springs (ASP)
- Launceston (LST)
- Rockhampton (ROK)

## 📊 Analysis Features

### **Price Trend Analysis**
- **Statistical Analysis**: Mean, median, standard deviation
- **Trend Detection**: Price volatility and daily changes
- **Demand Patterns**: High/low demand day identification
- **Visualizations**: Interactive line charts, histograms, heatmaps

### **Route Comparison**
- **Popular Routes**: Pre-defined popular route combinations
- **Price Rankings**: Most affordable vs expensive routes
- **Volatility Analysis**: Price stability across routes
- **Market Insights**: Route-specific recommendations

### **AI-Powered Insights**
- **Booking Recommendations**: Optimal booking timing
- **Cost-Saving Tips**: Price trend analysis
- **Market Dynamics**: Demand fluctuation insights
- **Actionable Advice**: Specific recommendations for travelers

## 🎯 Use Cases

### **For Travelers**
- Find the best time to book flights
- Compare prices across different routes
- Identify cost-saving opportunities
- Get AI-powered booking recommendations

### **For Travel Agencies**
- Analyze market trends
- Identify high-demand periods
- Optimize pricing strategies
- Understand route profitability

### **For Airlines**
- Monitor competitor pricing
- Identify market opportunities
- Analyze demand patterns
- Optimize route planning

## 📈 Key Metrics Analyzed

- **Average Price**: Mean cost across analyzed period
- **Price Range**: Difference between min and max prices
- **Price Volatility**: Standard deviation of price changes
- **Demand Patterns**: High/low demand day identification
- **Route Rankings**: Comparative analysis across routes
- **Booking Recommendations**: AI-generated insights

## 🔮 Future Enhancements

- **Seasonal Analysis**: Year-round trend analysis
- **Demand Forecasting**: AI-powered price predictions
- **Multi-Currency Support**: International route analysis
- **Real-time Alerts**: Price drop notifications
- **Mobile App**: Native mobile application
- **API Integration**: Additional data sources

## 🛡️ Rate Limiting

The application includes built-in rate limiting to respect API limits:
- 0.5-second delays between API calls
- Progress indicators for long-running analyses
- Error handling for API failures

## 📝 Usage Examples

### Single Route Analysis
1. Select departure and arrival cities
2. Choose start date and analysis period (7-30 days)
3. Click "Analyze Route" to get comprehensive insights
4. Review price trends, demand patterns, and AI recommendations

### Popular Routes Comparison
1. Navigate to "Popular Routes Comparison"
2. Click "Analyze Popular Routes"
3. Review route rankings and market insights
4. Compare prices across different city pairs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the documentation
- Review API key configuration
- Ensure all dependencies are installed
- Verify internet connectivity for API calls

---

**Built with ❤️ using Streamlit, SerpAPI, and DeepSeek AI** 