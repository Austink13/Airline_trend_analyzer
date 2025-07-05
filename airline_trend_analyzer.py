import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import time

# Import configuration
from config import (
    SERPAPI_KEY, DEEPSEEK_API_KEY, IATA_CODES, POPULAR_ROUTES,
    CURRENCY, DEFAULT_ANALYSIS_DAYS, MAX_ANALYSIS_DAYS, RATE_LIMIT_DELAY,
    PRICE_THRESHOLDS, CHART_HEIGHT, CHART_COLORS, DEEPSEEK_CONFIG
)

# Use configuration variables
iata_codes = IATA_CODES
popular_routes = POPULAR_ROUTES

# Custom CSS for modern UI with fixed sidebar
def load_custom_css():
    st.markdown("""
    <style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .main-header h2 {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .main-header h3 {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #7f8c8d;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    
    /* FIXED SIDEBAR STYLING */
    /* Target the sidebar container */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
    }
    
    /* Streamlit sidebar content */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
    }
    
    /* All text elements in sidebar - BLACK COLOR */
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Specific elements for better targeting */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stDateInput label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stButton label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #000000 !important;
    }
    
    /* Selectbox dropdown styling */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div > div {
        color: #000000 !important;
    }
    
    /* Checkbox styling */
    section[data-testid="stSidebar"] .stCheckbox > label {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] .stCheckbox > label > div {
        color: #000000 !important;
    }
    
    /* Slider styling */
    section[data-testid="stSidebar"] .stSlider > label {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] .stSlider > div {
        color: #000000 !important;
    }
    
    /* Date input styling */
    section[data-testid="stSidebar"] .stDateInput > label {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] .stDateInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Button styling in sidebar */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Markdown in sidebar */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4,
    section[data-testid="stSidebar"] .stMarkdown h5,
    section[data-testid="stSidebar"] .stMarkdown h6,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown div {
        color: #000000 !important;
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
    }
    
    /* Main content button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Main content selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e1e8ed;
        background-color: white;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    
    /* Main content selectbox labels - BLACK COLOR */
    .stSelectbox > label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* All selectbox labels in main content */
    .stSelectbox label,
    .stSelectbox p,
    .stSelectbox div {
        color: #000000 !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chart container styling */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Error message styling */
    .success-message {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Loading animation */
    .loading-container {
        text-align: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create a beautiful header section"""
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Airline Market Trend Analyzer</h1>
        <p>Advanced analytics for airline booking trends, price analysis, and market insights</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, subtitle="", icon="📊"):
    """Create a beautiful metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <h3 style="margin: 0; color: #2c3e50; font-size: 1.1rem;">{title}</h3>
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: #667eea; margin-bottom: 0.25rem;">
            {value}
        </div>
        {f'<div style="color: #7f8c8d; font-size: 0.9rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def create_insight_card(title, content, icon="💡"):
    """Create a beautiful insight card"""
    st.markdown(f"""
    <div class="insight-card">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <h3 style="margin: 0; color: white; font-size: 1.2rem;">{title}</h3>
        </div>
        <div style="color: white; line-height: 1.6;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_loading_animation(message="Analyzing data..."):
    """Create a beautiful loading animation"""
    st.markdown(f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <p style="margin-top: 1rem; color: #667eea; font-weight: 600;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def get_flight_data(dep_code: str, arr_code: str, date: str) -> List[Dict]:
    """Fetch flight data from SerpAPI"""
    params = {
        "engine": "google_flights",
        "departure_id": dep_code,
        "arrival_id": arr_code,
        "outbound_date": date,
        "type": "2",
        "hl": "en",
        "currency": CURRENCY,
        "api_key": SERPAPI_KEY
    }
    
    try:
        r = requests.get("https://serpapi.com/search", params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data.get("best_flights", [])
        else:
            st.error(f"Failed to fetch data for {date}: HTTP {r.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching data for {date}: {str(e)}")
        return []

def extract_flight_details(flights: List[Dict]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Extract price, airline, and duration from flight data (robust for SerpAPI)"""
    if not flights:
        return None, None, None

    lowest_price = float('inf')
    best_airline = None
    best_duration = None

    for flight in flights:
        if isinstance(flight, dict):
            # Extract price
            price_info = flight.get("price", None)
            amount = None
            if isinstance(price_info, dict):
                amount = price_info.get("amount")
            elif isinstance(price_info, (int, float)):
                amount = price_info

            # Extract airline and duration from legs if available
            airline = "Unknown"
            duration = "Unknown"
            legs = flight.get("legs") or flight.get("flight_segments")
            if isinstance(legs, list) and len(legs) > 0:
                first_leg = legs[0]
                airline = first_leg.get("airline", airline)
                duration = first_leg.get("duration", duration)
            # Fallback to top-level keys if present
            airline = flight.get("airline", airline)
            duration = flight.get("duration", duration)

            if amount is not None and amount < lowest_price:
                lowest_price = amount
                best_airline = airline
                best_duration = duration

    return (lowest_price if lowest_price != float('inf') else None, best_airline, best_duration)

def analyze_price_trends(df: pd.DataFrame) -> Dict:
    """Analyze price trends and patterns"""
    analysis = {}
    
    if df.empty or df['price'].isna().all():
        return analysis
    
    # Remove NaN values for analysis
    clean_df = df.dropna(subset=['price'])
    
    if clean_df.empty:
        return analysis
    
    # Basic statistics
    analysis['mean_price'] = clean_df['price'].mean()
    analysis['median_price'] = clean_df['price'].median()
    analysis['std_price'] = clean_df['price'].std()
    analysis['min_price'] = clean_df['price'].min()
    analysis['max_price'] = clean_df['price'].max()
    analysis['price_range'] = analysis['max_price'] - analysis['min_price']
    
    # Trend analysis
    if len(clean_df) > 1:
        # Calculate price change
        price_changes = clean_df['price'].diff().dropna()
        analysis['avg_daily_change'] = price_changes.mean()
        analysis['price_volatility'] = price_changes.std()
        
        # Identify cheapest and most expensive days
        min_idx = clean_df['price'].idxmin()
        max_idx = clean_df['price'].idxmax()
        analysis['cheapest_day'] = clean_df.loc[min_idx, 'date']
        analysis['most_expensive_day'] = clean_df.loc[max_idx, 'date']
        
        # Detect demand patterns
        price_threshold = analysis['mean_price'] * PRICE_THRESHOLDS['low_demand']
        low_demand_days = clean_df[clean_df['price'] <= price_threshold]['date'].tolist()
        high_demand_days = clean_df[clean_df['price'] >= analysis['mean_price'] * PRICE_THRESHOLDS['high_demand']]['date'].tolist()
        
        analysis['low_demand_days'] = low_demand_days
        analysis['high_demand_days'] = high_demand_days
    
    return analysis

def get_deepseek_insights(data: Dict, route: str) -> str:
    """Get AI-powered insights using DeepSeek API with fallback to local analysis"""
    if not DEEPSEEK_API_KEY or not DEEPSEEK_CONFIG["enabled"]:
        return generate_local_insights(data, route)
    
    prompt = f"""
    Analyze this airline booking data for the route {route}:
    
    Price Statistics:
    - Mean Price: ${data.get('mean_price', 'N/A'):.2f}
    - Median Price: ${data.get('median_price', 'N/A'):.2f}
    - Price Range: ${data.get('price_range', 'N/A'):.2f}
    - Average Daily Change: ${data.get('avg_daily_change', 'N/A'):.2f}
    
    Demand Patterns:
    - Cheapest Day: {data.get('cheapest_day', 'N/A')}
    - Most Expensive Day: {data.get('most_expensive_day', 'N/A')}
    - Low Demand Days: {data.get('low_demand_days', [])}
    - High Demand Days: {data.get('high_demand_days', [])}
    
    Provide actionable insights about:
    1. Best booking timing
    2. Price trends and patterns
    3. Demand fluctuations
    4. Cost-saving recommendations
    5. Market dynamics
    
    Format the response as clear, actionable insights with specific recommendations.
    """
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": DEEPSEEK_CONFIG["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": DEEPSEEK_CONFIG["temperature"],
            "max_tokens": DEEPSEEK_CONFIG["max_tokens"]
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=DEEPSEEK_CONFIG["timeout"]
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        elif response.status_code == 402:
            st.warning("⚠️ DeepSeek API usage limit reached. Using local analysis instead.")
            return generate_local_insights(data, route)
        else:
            st.warning(f"⚠️ DeepSeek API error (HTTP {response.status_code}). Using local analysis instead.")
            return generate_local_insights(data, route)
            
    except Exception as e:
        st.warning(f"⚠️ Error connecting to DeepSeek API: {str(e)}. Using local analysis instead.")
        return generate_local_insights(data, route)

def generate_local_insights(data: Dict, route: str) -> str:
    """Generate insights locally when DeepSeek API is unavailable"""
    
    insights = []
    
    # Price analysis
    if 'mean_price' in data and data['mean_price']:
        insights.append(f"💰 **Price Analysis for {route}:**")
        insights.append(f"• Average price: ${data['mean_price']:.2f}")
        
        if 'price_range' in data and data['price_range']:
            insights.append(f"• Price range: ${data['price_range']:.2f}")
            
            if data['price_range'] > data['mean_price'] * 0.3:
                insights.append("• High price volatility detected - consider flexible booking dates")
            else:
                insights.append("• Stable pricing - good for advance booking")
    
    # Booking recommendations
    if 'cheapest_day' in data and data['cheapest_day'] != 'N/A':
        insights.append(f"\n🎯 **Best Booking Strategy:**")
        insights.append(f"• Cheapest day: {data['cheapest_day']}")
        insights.append(f"• Most expensive day: {data['most_expensive_day']}")
        
        if 'low_demand_days' in data and data['low_demand_days']:
            insights.append(f"• Low demand days: {', '.join(data['low_demand_days'][:3])}")
            insights.append("• Consider booking on these days for better prices")
    
    # Trend analysis
    if 'avg_daily_change' in data and data['avg_daily_change']:
        insights.append(f"\n📈 **Trend Analysis:**")
        if data['avg_daily_change'] > 0:
            insights.append("• Prices are trending upward - book soon")
        elif data['avg_daily_change'] < 0:
            insights.append("• Prices are trending downward - consider waiting")
        else:
            insights.append("• Prices are relatively stable")
    
    # Demand patterns
    if 'high_demand_days' in data and data['high_demand_days']:
        insights.append(f"\n📊 **Demand Patterns:**")
        insights.append(f"• High demand detected on {len(data['high_demand_days'])} days")
        insights.append("• Avoid these dates for better prices")
    
    # Cost-saving tips
    insights.append(f"\n💡 **Cost-Saving Recommendations:**")
    insights.append("• Book 2-3 weeks in advance for best prices")
    insights.append("• Consider flexible dates around the cheapest day")
    insights.append("• Monitor prices daily for sudden drops")
    insights.append("• Check multiple airlines for competitive rates")
    
    # Market dynamics
    insights.append(f"\n🌍 **Market Dynamics:**")
    insights.append("• Australian domestic routes typically have seasonal patterns")
    insights.append("• Business travel affects weekday pricing")
    insights.append("• School holidays impact family travel demand")
    
    return "\n".join(insights)

def create_advanced_visualizations(df: pd.DataFrame, analysis: Dict, route: str):
    """Create comprehensive visualizations with enhanced styling"""
    
    # Price Trend Chart
    fig_trend = go.Figure()
    
    if not df.empty and not df['price'].isna().all():
        fig_trend.add_trace(go.Scatter(
            x=df['date'],
            y=df['price'],
            mode='lines+markers',
            name='Price Trend',
            line=dict(color=CHART_COLORS["primary"], width=4),
            marker=dict(size=10, color=CHART_COLORS["primary"])
        ))
        
        # Add mean price line
        if 'mean_price' in analysis:
            fig_trend.add_hline(
                y=analysis['mean_price'],
                line_dash="dash",
                line_color=CHART_COLORS["danger"],
                annotation_text=f"Mean Price: ${analysis['mean_price']:.2f}"
            )
    
    fig_trend.update_layout(
        title=dict(
            text=f"Price Trend for {route}",
            font=dict(size=20, color="#2c3e50")
        ),
        xaxis_title="Date",
        yaxis_title=f"Price ({CURRENCY})",
        hovermode='x unified',
        height=CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#2c3e50"),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Price Distribution
    if not df.empty and not df['price'].isna().all():
        fig_dist = px.histogram(
            df.dropna(subset=['price']),
            x='price',
            nbins=10,
            title=f"Price Distribution for {route}",
            labels={'price': f'Price ({CURRENCY})', 'count': 'Frequency'},
            color_discrete_sequence=[CHART_COLORS["primary"]]
        )
        
        fig_dist.update_layout(
            title=dict(font=dict(size=20, color="#2c3e50")),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#2c3e50"),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Demand Heatmap
    if not df.empty:
        df_heatmap = df.copy()
        df_heatmap['day_of_week'] = pd.to_datetime(df_heatmap['date']).dt.day_name()
        df_heatmap['price_category'] = pd.cut(
            df_heatmap['price'], 
            bins=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        heatmap_data = df_heatmap.groupby(['day_of_week', 'price_category']).size().unstack(fill_value=0)
        
        fig_heatmap = px.imshow(
            heatmap_data,
            title=f"Demand Heatmap for {route}",
            labels=dict(x="Price Category", y="Day of Week", color="Frequency"),
            color_continuous_scale="Viridis"
        )
        
        fig_heatmap.update_layout(
            title=dict(font=dict(size=20, color="#2c3e50")),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#2c3e50"),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def analyze_popular_routes():
    """Analyze popular routes for market insights with enhanced UI"""
    st.markdown("""
    <div class="main-header">
        <h2>🏆 Popular Routes Analysis</h2>
        <p>Compare prices and trends across popular Australian routes</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Analyze Popular Routes", use_container_width=True):
            with st.spinner(""):
                create_loading_animation("Analyzing popular routes...")
                
                route_data = []
                
                for dep, arr in popular_routes[:8]:  # Limit to 8 routes for performance
                    dep_code = iata_codes[dep]
                    arr_code = iata_codes[arr]
                    route_name = f"{dep} → {arr}"
                    
                    # Get data for next 7 days
                    prices = []
                    for i in range(7):
                        date = datetime.today() + timedelta(days=i)
                        date_str = date.strftime("%Y-%m-%d")
                        flights = get_flight_data(dep_code, arr_code, date_str)
                        price, _, _ = extract_flight_details(flights)
                        if price:
                            prices.append(price)
                        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                    
                    if prices:
                        route_data.append({
                            'route': route_name,
                            'avg_price': np.mean(prices),
                            'min_price': min(prices),
                            'max_price': max(prices),
                            'price_volatility': np.std(prices),
                            'sample_size': len(prices)
                        })
                
                if route_data:
                    routes_df = pd.DataFrame(route_data)
                    
                    # Route comparison chart
                    fig_routes = px.bar(
                        routes_df,
                        x='route',
                        y='avg_price',
                        title="Average Prices by Route",
                        labels={'avg_price': f'Average Price ({CURRENCY})', 'route': 'Route'},
                        color='price_volatility',
                        color_continuous_scale='viridis'
                    )
                    
                    fig_routes.update_layout(
                        title=dict(font=dict(size=20, color="#2c3e50")),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#2c3e50"),
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                    
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(fig_routes, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Route ranking with enhanced UI
                    st.markdown("""
                    <div class="main-header">
                        <h3>📊 Route Rankings</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("**💰 Most Expensive Routes:**")
                        expensive_routes = routes_df.nlargest(3, 'avg_price')
                        for _, route in expensive_routes.iterrows():
                            st.markdown(f"• **{route['route']}**: ${route['avg_price']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown("**💎 Most Affordable Routes:**")
                        affordable_routes = routes_df.nsmallest(3, 'avg_price')
                        for _, route in affordable_routes.iterrows():
                            st.markdown(f"• **{route['route']}**: ${route['avg_price']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced dataframe display
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("**📋 Detailed Route Data**")
                    st.dataframe(routes_df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Airline Market Trend Analyzer",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Create header
    create_header()
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">📊 Analysis Options</h3>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_type = st.selectbox(
            "Choose Analysis Type",
            ["Single Route Analysis", "Popular Routes Comparison", "Market Insights"],
            help="Select the type of analysis you want to perform"
        )
        
        st.markdown("---")
        
        # AI Insights toggle
        ai_enabled = st.checkbox(
            "🤖 Enable AI Insights", 
            value=DEEPSEEK_CONFIG["enabled"],
            help="Enable DeepSeek AI for advanced insights (requires API key)"
        )
        
        # Update config based on user choice
        if ai_enabled != DEEPSEEK_CONFIG["enabled"]:
            DEEPSEEK_CONFIG["enabled"] = ai_enabled
        
        st.markdown("---")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;">
            <h4 style="color: white; margin-bottom: 0.5rem;">ℹ️ About</h4>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0;">
                This app analyzes airline booking trends using Google Flights data and AI-powered insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if analysis_type == "Single Route Analysis":
        st.markdown("""
        <div class="main-header">
            <h2>🔍 Single Route Analysis</h2>
            <p>Analyze price trends and demand patterns for specific routes</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced form layout
        col1, col2 = st.columns(2)
        with col1:
            departure = st.selectbox("🛫 Departure City", list(iata_codes.keys()))
        with col2:
            arrival = st.selectbox("🛬 Arrival City", list(iata_codes.keys()), index=1)
        
        col1, col2 = st.columns(2)
        with col1:
            travel_date = st.date_input(
                "📅 Start Travel Date", 
                value=datetime.today(), 
                min_value=datetime.today()
            )
        with col2:
            days_to_analyze = st.slider("📊 Days to Analyze", 7, MAX_ANALYSIS_DAYS, DEFAULT_ANALYSIS_DAYS)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Route", use_container_width=True):
                if departure == arrival:
                    st.error("Departure and arrival cities cannot be the same.")
                else:
                    with st.spinner(""):
                        create_loading_animation(f"Analyzing {departure} → {arrival} route...")
                        
                        dep_code = iata_codes[departure]
                        arr_code = iata_codes[arrival]
                        route_name = f"{departure} → {arrival}"
                        
                        trend_data = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(days_to_analyze):
                            date = travel_date + timedelta(days=i)
                            date_str = date.strftime("%Y-%m-%d")
                            flights = get_flight_data(dep_code, arr_code, date_str)
                            if not flights:
                                price, airline, duration = None, 'No flights found', 'No flights found'
                            else:
                                price, airline, duration = extract_flight_details(flights)
                            
                            trend_data.append({
                                "date": date_str,
                                "price": price,
                                "airline": airline,
                                "duration": duration
                            })
                            
                            # Update progress
                            progress = (i + 1) / days_to_analyze
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {i + 1}/{days_to_analyze} days...")
                            
                            time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                        
                        df = pd.DataFrame(trend_data)
                        
                        if not df.empty:
                            # Analysis
                            analysis = analyze_price_trends(df)
                            
                            # Display insights with enhanced UI
                            st.markdown("""
                            <div class="main-header">
                                <h3>📈 Analysis Results</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics in cards
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                create_metric_card("Average Price", f"${analysis.get('mean_price', 0):.2f}", "Mean cost", "💰")
                            with col2:
                                create_metric_card("Price Range", f"${analysis.get('price_range', 0):.2f}", "Min to Max", "📊")
                            with col3:
                                create_metric_card("Cheapest Day", analysis.get('cheapest_day', 'N/A'), "Best deal", "🎯")
                            with col4:
                                create_metric_card("Most Expensive", analysis.get('most_expensive_day', 'N/A'), "Peak price", "⚠️")
                            
                            # Visualizations
                            create_advanced_visualizations(df, analysis, route_name)
                            
                            # AI Insights with enhanced styling
                            st.markdown("""
                            <div class="main-header">
                                <h3>🤖 AI-Powered Insights</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            insights = get_deepseek_insights(analysis, route_name)
                            create_insight_card("Market Intelligence", insights, "🧠")
                            
                            # Detailed data with enhanced styling
                            st.markdown("""
                            <div class="main-header">
                                <h3>📋 Detailed Data</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.dataframe(df, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("No data available for this route.")
    
    elif analysis_type == "Popular Routes Comparison":
        analyze_popular_routes()
    
    elif analysis_type == "Market Insights":
        st.markdown("""
        <div class="main-header">
            <h2>📈 Market Insights</h2>
            <p>Comprehensive market analysis and trends</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Market overview with enhanced cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card("Total Routes", len(popular_routes), "Analyzed routes", "🛣️")
        with col2:
            create_metric_card("Cities Covered", len(iata_codes), "Australian cities", "🏙️")
        with col3:
            create_metric_card("Data Points", len(popular_routes) * 7, "Total samples", "📊")
        
        # Coming soon sections with enhanced styling
        st.markdown("""
        <div class="main-header">
            <h3>🔄 Coming Soon</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_insight_card("Seasonal Trends", "Advanced seasonal analysis with year-round trend detection and holiday impact analysis.", "📅")
        
        with col2:
            create_insight_card("Demand Forecasting", "AI-powered demand forecasting with predictive analytics and price prediction models.", "🔮")

if __name__ == "__main__":
    main() 