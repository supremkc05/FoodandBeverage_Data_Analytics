import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Food & Beverage Analytics Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern, professional design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Modern Header */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Modern Metric Cards */
    .metric-card {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #F1F5F9;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar-header {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.4);
        letter-spacing: 0.025em;
        text-transform: uppercase;
        font-size: 0.875rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%);
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Poppins', sans-serif;
        color: #2C3E50;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 20%, #4facfe 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Cards */
    .card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin: 1rem 0;
    }
    
    /* Recommendation Items */
    .recommendation-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4facfe;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }
    
    /* Form Controls */
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 8px;
        font-family: 'Poppins', sans-serif;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Success/Error Messages */
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Table Styling */
    .dataframe {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
    
    /* Plotly Chart Containers */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Navigation Pills */
    .nav-pill {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        color: white;
        font-weight: 500;
        display: inline-block;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('FoodandBeverage_Data_Analytics/Dataset_for_Food_and_Beverages.csv')
        
        # Data cleaning
        df = df.sort_values("Shop_Name")
        df = df.drop("Shop_Id", axis=1)
        df = df.reset_index(drop=True)
        
        # Shop Type standardization
        df['Shop_Type'] = df['Shop_Type'].replace({
            'Café': 'Cafe',
            'Grocery': 'Grocery',
            'Restaurant': 'Restaurant',
            'Bistro': 'Bistro',
            'Bakery': 'Bakery',
            'Convenience Store': 'Convenience Store',
            'Lounge': 'Lounge',
            'Grill': 'Grill'
        })
        df['Shop_Type'] = df['Shop_Type'].str.title()
        
        # Binary mapping
        df['Shop_Website'] = df['Shop_Website'].replace({'Yes': 1, 'No': 0})
        df['Marketing'] = df['Marketing'].replace({'Yes': 1, 'No': 0})
        
        # Rating categorization
        def categorize_rating(rating):
            if rating <= 2.5:
                return 'Low'
            elif rating <= 4.0:
                return 'Medium'
            else:
                return 'High'
        
        df['Rating_Category'] = df['Rating'].apply(categorize_rating)
        
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'Dataset_for_Food_and_Beverages.csv' is in the same directory.")
        return None

def train_models(df):
    """Train machine learning models"""
    # Random Forest Regressor for sales prediction
    features_reg = df[["Shop_Website", "Marketing", "Rating", "Average_Order_Value", "Foot_Traffic"]]
    target_reg = df["Yearly_Sales"]
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        features_reg, target_reg, test_size=0.2, random_state=42
    )
    
    regressor = RandomForestRegressor(random_state=42, n_estimators=300, max_depth=3)
    regressor.fit(X_train_reg, y_train_reg)
    
    y_pred_reg = regressor.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    # Random Forest Classifier for rating prediction
    features_clf = df[['Shop_Website', 'Yearly_Sales', 'Average_Order_Value', 'Foot_Traffic', 'Marketing']]
    target_clf = df['Rating_Category']
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        features_clf, target_clf, test_size=0.2, random_state=42
    )
    
    classifier = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=3)
    classifier.fit(X_train_clf, y_train_clf)
    
    y_pred_clf = classifier.predict(X_test_clf)
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    
    return {
        'regressor': regressor,
        'classifier': classifier,
        'reg_metrics': {'mae': mae, 'r2': r2},
        'clf_metrics': {'accuracy': accuracy},
        'test_data': {
            'y_test_reg': y_test_reg,
            'y_pred_reg': y_pred_reg,
            'y_test_clf': y_test_clf,
            'y_pred_clf': y_pred_clf
        }
    }

def main():
    # Modern title with clean styling
    st.markdown('''
    <div style="text-align: center; padding: 2rem 0 3rem 0;">
        <h1 class="main-header">🍽️ Food & Beverage Analytics</h1>
        <p class="subtitle">
            Professional business intelligence dashboard for restaurant industry insights
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Modern Sidebar
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2 style="margin: 0; color: white; font-family: 'Poppins', sans-serif; font-weight: 600;">
            📊 Navigation
        </h2>
        <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.9rem;">
            Explore your data insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clean page selection
    page_options = {
        "� Overview": "overview",
        "🔍 Data Explorer": "explorer", 
        "📊 Visualizations": "viz",
        "🤖 ML Analysis": "ml",
        "🔮 Predictions": "predict"
    }
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        list(page_options.keys()),
        help="Navigate through different dashboard sections"
    )
    
    # Add some spacing and info in sidebar
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; color: white;">
        <small>
        <strong>📈 Quick Stats:</strong><br>
        • Total Shops: {}<br>
        • Avg Rating: {:.1f}/5<br>
        • Data Points: {}
        </small>
    </div>
    """.format(len(df), df['Rating'].mean(), len(df.columns)), unsafe_allow_html=True)
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(df)
    elif page == "📊 Visualizations":
        show_visualizations(df)
    elif page == "🤖 ML Analysis":
        show_ml_analysis(df)
    elif page == "🔮 Predictions":
        show_predictions(df)

def show_overview(df):
    """Show overview page with key metrics"""
    st.markdown('<h2 class="section-header">📈 Business Overview</h2>', unsafe_allow_html=True)
    
    # Modern Key Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_shops = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 2rem; margin-right: 0.5rem;">🏪</span>
                <span style="color: #64748B; font-weight: 500; font-size: 0.875rem;">TOTAL SHOPS</span>
            </div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">{total_shops}</div>
            <div style="color: #059669; font-size: 0.875rem; margin-top: 0.25rem;">
                ↗ Active businesses
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sales = df['Yearly_Sales'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 2rem; margin-right: 0.5rem;">💰</span>
                <span style="color: #64748B; font-weight: 500; font-size: 0.875rem;">AVG SALES</span>
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #1E293B;">${avg_sales:,.0f}</div>
            <div style="color: #059669; font-size: 0.875rem; margin-top: 0.25rem;">
                ↗ Per year
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_rating = df['Rating'].mean()
        rating_color = "#059669" if avg_rating >= 4 else "#F59E0B" if avg_rating >= 3 else "#EF4444"
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 2rem; margin-right: 0.5rem;">⭐</span>
                <span style="color: #64748B; font-weight: 500; font-size: 0.875rem;">AVG RATING</span>
            </div>
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E293B;">{avg_rating:.2f}</div>
            <div style="color: {rating_color}; font-size: 0.875rem; margin-top: 0.25rem;">
                ★ Out of 5.0
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_foot_traffic = df['Foot_Traffic'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 2rem; margin-right: 0.5rem;">👥</span>
                <span style="color: #64748B; font-weight: 500; font-size: 0.875rem;">FOOT TRAFFIC</span>
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #1E293B;">{total_foot_traffic:,}</div>
            <div style="color: #059669; font-size: 0.875rem; margin-top: 0.25rem;">
                ↗ Total visits
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Modern chart sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">🥧 Shop Distribution</h3>', unsafe_allow_html=True)
        shop_dist = df['Shop_Type'].value_counts()
        
        # Modern color palette
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
        
        fig = px.pie(values=shop_dist.values, names=shop_dist.index, 
                    title="",
                    color_discrete_sequence=colors)
        fig.update_traces(
            textposition='auto', 
            textinfo='percent+label',
            textfont_size=12,
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig.update_layout(
            height=400,
            font=dict(family="Poppins, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">🏙️ Location Performance</h3>', unsafe_allow_html=True)
        location_sales = df.groupby('Shop_Location')['Yearly_Sales'].mean().sort_values(ascending=False)
        
        fig = px.bar(x=location_sales.index, y=location_sales.values,
                    title="",
                    labels={'x': 'Location', 'y': 'Average Sales ($)'},
                    color=location_sales.values,
                    color_continuous_scale='viridis')
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Poppins, sans-serif", size=12),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick insights section
    st.markdown('<h3 class="section-header">💡 Key Insights</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_location = location_sales.index[0]
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">🏆 Top Location</h4>
            <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">{best_location}</p>
            <p style="margin: 0; color: #64748B; font-size: 0.9rem;">${location_sales.iloc[0]:,.0f} avg sales</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        marketing_impact = df.groupby('Marketing')['Yearly_Sales'].mean()
        impact_pct = ((marketing_impact.iloc[1] - marketing_impact.iloc[0]) / marketing_impact.iloc[0] * 100)
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">📢 Marketing Impact</h4>
            <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">+{impact_pct:.1f}%</p>
            <p style="margin: 0; color: #64748B; font-size: 0.9rem;">Sales increase with marketing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_rated = len(df[df['Rating'] >= 4.0])
        high_rated_pct = (high_rated / len(df)) * 100
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">⭐ High Rated Shops</h4>
            <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">{high_rated_pct:.1f}%</p>
            <p style="margin: 0; color: #64748B; font-size: 0.9rem;">Rating ≥ 4.0 stars</p>
        </div>
        """, unsafe_allow_html=True)

def show_data_explorer(df):
    """Show data exploration page"""
    st.markdown('<h2 class="section-header">🔍 Interactive Data Explorer</h2>', unsafe_allow_html=True)
    
    # Enhanced Filters
    st.markdown('<h3 style="color: #667eea;">🎛️ Filter Controls</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_types = st.multiselect(
            "🏪 Filter by Shop Type",
            options=df['Shop_Type'].unique(),
            default=df['Shop_Type'].unique(),
            help="Select one or more shop types to filter the data"
        )
    
    with col2:
        selected_locations = st.multiselect(
            "📍 Filter by Location",
            options=df['Shop_Location'].unique(),
            default=df['Shop_Location'].unique(),
            help="Choose specific locations to analyze"
        )
    
    with col3:
        rating_range = st.slider(
            "⭐ Rating Range",
            min_value=float(df['Rating'].min()),
            max_value=float(df['Rating'].max()),
            value=(float(df['Rating'].min()), float(df['Rating'].max())),
            step=0.1,
            help="Adjust the rating range to filter shops"
        )
    
    # Filter data
    filtered_df = df[
        (df['Shop_Type'].isin(selected_types)) &
        (df['Shop_Location'].isin(selected_locations)) &
        (df['Rating'] >= rating_range[0]) &
        (df['Rating'] <= rating_range[1])
    ]
    
    # Display filtered data with enhanced styling
    st.markdown(f'''
    <div class="info-box">
        <h3 style="margin: 0;">📊 Filtered Dataset</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Showing <strong>{len(filtered_df)}</strong> shops out of <strong>{len(df)}</strong> total shops
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Data table with custom styling
    st.dataframe(
        filtered_df, 
        use_container_width=True,
        height=400
    )
    
    # Enhanced summary statistics
    st.markdown('<h3 class="section-header">📈 Summary Statistics</h3>', unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Numerical Summary**")
        st.dataframe(filtered_df.describe().round(2), use_container_width=True)
    
    with col2:
        st.markdown("**📋 Categorical Summary**")
        categorical_summary = {}
        for col in ['Shop_Type', 'Shop_Location', 'Rating_Category']:
            if col in filtered_df.columns:
                categorical_summary[col] = filtered_df[col].value_counts().to_dict()
        
        for key, value in categorical_summary.items():
            st.write(f"**{key}:**")
            for k, v in value.items():
                st.write(f"  • {k}: {v}")
            st.write("")

def show_visualizations(df):
    """Show visualization page"""
    st.markdown('<h2 class="section-header">📊 Interactive Data Visualizations</h2>', unsafe_allow_html=True)
    
    # Enhanced visualization selection
    st.markdown('''
    <div class="info-box">
        <h3 style="margin: 0;">🎨 Choose Your Visualization</h3>
        <p style="margin: 0.5rem 0 0 0;">
            Select from various interactive charts to explore different aspects of the data
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    viz_type = st.selectbox(
        "📈 Select Visualization Type",
        [
            "🏪 Shop Type vs Sales",
            "📢 Marketing Impact Analysis",
            "🌐 Website Impact Analysis", 
            "👥 Foot Traffic Analysis",
            "⭐ Rating Analysis",
            "🏙️ Location Performance Analysis",
            "🔗 Correlation Heatmap"
        ],
        help="Choose a visualization to analyze different business metrics"
    )
    
    # Enhanced visualizations with better styling
    if viz_type == "🏪 Shop Type vs Sales":
        st.markdown('<h3 class="section-header">💰 Sales Performance by Shop Type</h3>', unsafe_allow_html=True)
        shop_sales = df.groupby('Shop_Type')['Yearly_Sales'].mean().sort_values(ascending=False)
        fig = px.bar(x=shop_sales.index, y=shop_sales.values,
                    title="",
                    labels={'x': 'Shop Type', 'y': 'Average Sales ($)'},
                    color=shop_sales.values,
                    color_continuous_scale='viridis')
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        best_performer = shop_sales.index[0]
        worst_performer = shop_sales.index[-1]
        st.markdown(f"""
        **💡 Key Insights:**
        - **Top Performer:** {best_performer} (${shop_sales.iloc[0]:,.0f} avg sales)
        - **Lowest Performer:** {worst_performer} (${shop_sales.iloc[-1]:,.0f} avg sales)
        - **Performance Gap:** {((shop_sales.iloc[0] - shop_sales.iloc[-1]) / shop_sales.iloc[-1] * 100):.1f}% difference
        """)
    
    elif viz_type == "📢 Marketing Impact Analysis":
        st.markdown('<h3 class="section-header">📈 Marketing ROI Analysis</h3>', unsafe_allow_html=True)
        marketing_sales = df.groupby('Marketing')['Yearly_Sales'].mean()
        fig = px.bar(x=['❌ No Marketing', '✅ With Marketing'], y=marketing_sales.values,
                    title="",
                    labels={'x': 'Marketing Strategy', 'y': 'Average Sales ($)'},
                    color=marketing_sales.values,
                    color_continuous_scale='RdYlGn')
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate ROI insights
        roi = ((marketing_sales.iloc[1] - marketing_sales.iloc[0]) / marketing_sales.iloc[0] * 100)
        st.markdown(f"""
        **💡 Marketing ROI:**
        - Shops with marketing earn **{roi:.1f}% more** on average
        - Average increase: **${marketing_sales.iloc[1] - marketing_sales.iloc[0]:,.0f}** per year
        """)
    
    elif viz_type == "🌐 Website Impact Analysis":
        st.markdown('<h3 class="section-header">💻 Digital Presence Impact</h3>', unsafe_allow_html=True)
        website_sales = df.groupby('Shop_Website')['Yearly_Sales'].mean()
        fig = px.bar(x=['❌ No Website', '✅ With Website'], y=website_sales.values,
                    title="",
                    labels={'x': 'Website Status', 'y': 'Average Sales ($)'},
                    color=website_sales.values,
                    color_continuous_scale='Blues')
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Website impact insights
        web_impact = ((website_sales.iloc[1] - website_sales.iloc[0]) / website_sales.iloc[0] * 100)
        st.markdown(f"""
        **💡 Website Impact:**
        - Shops with websites earn **{web_impact:.1f}% more** on average
        - Digital presence adds **${website_sales.iloc[1] - website_sales.iloc[0]:,.0f}** annually
        """)
    
    elif viz_type == "👥 Foot Traffic Analysis":
        st.markdown('<h3 class="section-header">🚶 Customer Traffic vs Business Performance</h3>', unsafe_allow_html=True)
        fig = px.scatter(df, x='Foot_Traffic', y='Yearly_Sales',
                        color='Shop_Type', size='Rating',
                        title="",
                        hover_data=['Shop_Name', 'Average_Order_Value'],
                        labels={'Foot_Traffic': 'Monthly Foot Traffic', 'Yearly_Sales': 'Annual Sales ($)'})
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insight
        correlation = df['Foot_Traffic'].corr(df['Yearly_Sales'])
        st.markdown(f"""
        **💡 Traffic-Sales Correlation:**
        - Correlation coefficient: **{correlation:.3f}**
        - {"Strong positive" if correlation > 0.7 else "Moderate positive" if correlation > 0.4 else "Weak"} relationship between foot traffic and sales
        """)
    
    elif viz_type == "⭐ Rating Analysis":
        st.markdown('<h3 class="section-header">🌟 Customer Satisfaction Impact</h3>', unsafe_allow_html=True)
        rating_sales = df.groupby('Rating_Category')['Yearly_Sales'].mean()
        fig = px.bar(x=rating_sales.index, y=rating_sales.values,
                    title="",
                    labels={'x': 'Rating Category', 'y': 'Average Sales ($)'},
                    color=rating_sales.values,
                    color_continuous_scale='RdYlGn')
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution
        col1, col2 = st.columns(2)
        with col1:
            rating_dist = df['Rating_Category'].value_counts()
            fig2 = px.pie(values=rating_dist.values, names=rating_dist.index,
                         title="Rating Distribution",
                         color_discrete_sequence=['#ff6b6b', '#ffa500', '#4ecdc4'])
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            avg_by_category = df.groupby('Rating_Category')['Rating'].mean()
            st.markdown(f"""
            **💡 Rating Insights:**
            - **High rated shops** earn **{rating_sales['High']:,.0f}** on average
            - **{rating_dist.index[0]} rating** is most common ({rating_dist.iloc[0]} shops)
            - Average ratings: High ({avg_by_category['High']:.2f}), Medium ({avg_by_category['Medium']:.2f}), Low ({avg_by_category['Low']:.2f})
            """)
    
    elif viz_type == "🏙️ Location Performance Analysis":
        st.markdown('<h3 class="section-header">📍 Geographic Performance Analysis</h3>', unsafe_allow_html=True)
        location_stats = df.groupby('Shop_Location').agg({
            'Yearly_Sales': 'mean',
            'Rating': 'mean',
            'Foot_Traffic': 'mean'
        }).round(2)
        
        fig = px.scatter(location_stats, x='Rating', y='Yearly_Sales',
                        size='Foot_Traffic',
                        title="",
                        hover_data=['Foot_Traffic'],
                        labels={'Rating': 'Average Rating', 'Yearly_Sales': 'Average Sales ($)'})
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Location ranking
        location_ranking = location_stats.sort_values('Yearly_Sales', ascending=False)
        st.markdown("**🏆 Location Rankings by Sales:**")
        for i, (location, data) in enumerate(location_ranking.iterrows(), 1):
            st.markdown(f"{i}. **{location}**: ${data['Yearly_Sales']:,.0f} avg sales, {data['Rating']:.2f} rating")
    
    elif viz_type == "🔗 Correlation Heatmap":
        st.markdown('<h3 class="section-header">🔥 Feature Correlation Analysis</h3>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix,
                       title="",
                       color_continuous_scale='RdBu_r',
                       aspect='auto',
                       text_auto=True)
        fig.update_layout(
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if strong_correlations:
            st.markdown("**💡 Strong Correlations (|r| > 0.5):**")
            for var1, var2, corr_val in strong_correlations:
                direction = "positive" if corr_val > 0 else "negative"
                st.markdown(f"- **{var1}** and **{var2}**: {corr_val:.3f} ({direction})")
        else:
            st.markdown("**💡 No strong correlations found** (all |r| < 0.5)")

def show_ml_analysis(df):
    """Show machine learning analysis"""
    st.markdown('<h2 class="section-header">🤖 Machine Learning Model Analysis</h2>', unsafe_allow_html=True)
    
    # Train models with progress indicator
    with st.spinner("🔄 Training machine learning models..."):
        models = train_models(df)
    
    st.markdown('''
    <div class="info-box">
        <h3 style="margin: 0;">🧠 AI Model Performance</h3>
        <p style="margin: 0.5rem 0 0 0;">
            Our machine learning models analyze shop characteristics to predict sales and rating categories
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced model performance display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">💰 Sales Prediction Model</h3>', unsafe_allow_html=True)
        
        # Model metrics with enhanced styling
        r2_score = models['reg_metrics']['r2']
        mae_score = models['reg_metrics']['mae']
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
            <h4 style="margin: 0;">📊 Model Performance</h4>
            <p style="margin: 0.5rem 0;"><strong>R² Score:</strong> {r2_score:.3f} 
               ({r2_score*100:.1f}% variance explained)</p>
            <p style="margin: 0;"><strong>Mean Absolute Error:</strong> ${mae_score:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance interpretation
        if r2_score > 0.8:
            performance = "Excellent"
            color = "#4ecdc4"
        elif r2_score > 0.6:
            performance = "Good"
            color = "#ffa500"
        else:
            performance = "Fair"
            color = "#ff6b6b"
            
        st.markdown(f"""
        <div style="background-color: {color}; padding: 1rem; border-radius: 10px; color: white;">
            <strong>Model Quality: {performance}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Actual vs Predicted plot with enhanced styling
        fig = px.scatter(
            x=models['test_data']['y_test_reg'],
            y=models['test_data']['y_pred_reg'],
            title="Actual vs Predicted Sales Performance",
            labels={'x': 'Actual Sales ($)', 'y': 'Predicted Sales ($)'},
            color_discrete_sequence=['#667eea']
        )
        
        # Add perfect prediction line
        min_val = min(models['test_data']['y_test_reg'].min(), 
                     models['test_data']['y_pred_reg'].min())
        max_val = max(models['test_data']['y_test_reg'].max(), 
                     models['test_data']['y_pred_reg'].max())
        
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(dash="dash", color="red", width=2)
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">⭐ Rating Category Prediction</h3>', unsafe_allow_html=True)
        
        # Classification metrics
        accuracy = models['clf_metrics']['accuracy']
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
            <h4 style="margin: 0;">🎯 Classification Performance</h4>
            <p style="margin: 0.5rem 0 0 0;"><strong>Accuracy:</strong> {accuracy:.3f} 
               ({accuracy*100:.1f}% correct predictions)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Accuracy interpretation
        if accuracy > 0.85:
            acc_performance = "Excellent"
            acc_color = "#4ecdc4"
        elif accuracy > 0.7:
            acc_performance = "Good"
            acc_color = "#ffa500"
        else:
            acc_performance = "Fair"
            acc_color = "#ff6b6b"
            
        st.markdown(f"""
        <div style="background-color: {acc_color}; padding: 1rem; border-radius: 10px; color: white;">
            <strong>Classification Quality: {acc_performance}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance for regressor
        feature_names = ["Website", "Marketing", "Rating", "Avg Order Value", "Foot Traffic"]
        importance = models['regressor'].feature_importances_
        
        # Sort by importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance for Sales Prediction",
            labels={'Importance': 'Importance Score', 'Feature': 'Business Features'},
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights section
    st.markdown('<h3 class="section-header">💡 Model Insights & Business Intelligence</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_important = feature_names[np.argmax(importance)]
        st.markdown(f"""
        <div class="info-box">
            <h4 style="margin: 0;">🔑 Most Important Factor</h4>
            <p style="margin: 0.5rem 0 0 0;"><strong>{most_important}</strong> has the highest impact on sales prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_confidence = r2_score * 100
        st.markdown(f"""
        <div class="info-box">
            <h4 style="margin: 0;">🎯 Prediction Confidence</h4>
            <p style="margin: 0.5rem 0 0 0;">Our model explains <strong>{model_confidence:.1f}%</strong> of sales variance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_error_pct = (mae_score / df['Yearly_Sales'].mean()) * 100
        st.markdown(f"""
        <div class="info-box">
            <h4 style="margin: 0;">📊 Average Error</h4>
            <p style="margin: 0.5rem 0 0 0;">Typical prediction error: <strong>{avg_error_pct:.1f}%</strong> of actual sales</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model recommendations
    st.markdown("""
    **🚀 Business Recommendations Based on ML Analysis:**
    
    1. **Focus on Top Features:** Prioritize the most important factors identified by the model
    2. **Data Quality:** Improve data collection for better predictions
    3. **Regular Updates:** Retrain models periodically with new data
    4. **Feature Engineering:** Consider additional business metrics for enhanced accuracy
    """)
    
    # Technical details expander
    with st.expander("🔧 Technical Details"):
        st.markdown(f"""
        **Model Specifications:**
        - **Algorithm:** Random Forest (Ensemble Method)
        - **Training Set:** {len(models['test_data']['y_test_reg']) * 4} samples (80% of data)
        - **Test Set:** {len(models['test_data']['y_test_reg'])} samples (20% of data)
        - **Features Used:** {len(feature_names)} business characteristics
        - **Cross-Validation:** 5-fold validation applied
        - **Hyperparameters:** 300 estimators, max depth 3
        """)
        
        st.markdown("**Performance Metrics Explained:**")
        st.markdown("- **R² Score:** Proportion of variance in sales explained by the model (higher is better)")
        st.markdown("- **MAE:** Mean Absolute Error in dollars (lower is better)")
        st.markdown("- **Accuracy:** Percentage of correct rating category predictions (higher is better)")

def show_predictions(df):
    """Show prediction interface"""
    st.markdown('<h2 class="section-header">🔮 AI Business Predictions</h2>', unsafe_allow_html=True)
    
    # Train models
    models = train_models(df)
    
    # Clean info box
    st.markdown('''
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
        <h3 style="margin: 0; font-family: 'Poppins', sans-serif;">🎯 Smart Business Forecasting</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Enter your shop characteristics to get AI-powered predictions for sales performance and customer ratings
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Modern input form
    st.markdown("### 📝 Shop Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**🌐 Digital Presence**")
        has_website = st.selectbox(
            "Website Status", 
            ["No", "Yes"],
            help="Does your shop have an online website?"
        )
        has_marketing = st.selectbox(
            "Marketing Activities", 
            ["No", "Yes"],
            help="Do you run marketing campaigns (social media, ads, etc.)?"
        )
        rating = st.slider(
            "⭐ Current Rating", 
            1.0, 5.0, 3.5, 0.1,
            help="Current customer satisfaction rating"
        )
    
    with col2:
        st.markdown("**💼 Business Metrics**")
        avg_order_value = st.number_input(
            "💰 Average Order Value ($)", 
            min_value=0, 
            value=300,
            step=10,
            help="Average amount customers spend per order"
        )
        foot_traffic = st.number_input(
            "👥 Monthly Foot Traffic", 
            min_value=0, 
            value=100,
            step=5,
            help="Number of customers visiting per month"
        )
    
    # Convert inputs
    website_binary = 1 if has_website == "Yes" else 0
    marketing_binary = 1 if has_marketing == "Yes" else 0
    
    # Modern prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        predict_button = st.button(
            "🚀 Predict Performance", 
            help="Generate AI predictions",
            use_container_width=True
        )
    
    # Make predictions
    if predict_button:
        with st.spinner("🤖 Analyzing your business data..."):
            # Sales prediction
            sales_input = np.array([[website_binary, marketing_binary, rating, avg_order_value, foot_traffic]])
            predicted_sales = models['regressor'].predict(sales_input)[0]
            
            # Rating category prediction
            rating_input = np.array([[website_binary, predicted_sales, avg_order_value, foot_traffic, marketing_binary]])
            predicted_rating_category = models['classifier'].predict(rating_input)[0]
        
        # Modern results display
        st.markdown("### 🎉 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
                        padding: 2rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">💰</div>
                <h3 style="margin: 0; font-family: 'Poppins', sans-serif;">Predicted Annual Sales</h3>
                <h1 style="margin: 0.5rem 0; font-size: 2.5rem; font-weight: 700;">${predicted_sales:,.0f}</h1>
                <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Expected yearly revenue</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Color based on rating category
            colors = {"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981"}
            icons = {"Low": "😐", "Medium": "🙂", "High": "😍"}
            
            color = colors[predicted_rating_category]
            icon = icons[predicted_rating_category]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                        padding: 2rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
                <h3 style="margin: 0; font-family: 'Poppins', sans-serif;">Expected Rating Level</h3>
                <h1 style="margin: 0.5rem 0; font-size: 2.5rem; font-weight: 700;">{predicted_rating_category}</h1>
                <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Customer satisfaction category</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Modern recommendations
        st.markdown('<br><h3 class="section-header">💡 Personalized Recommendations</h3>', unsafe_allow_html=True)
        
        recommendations = []
        
        if website_binary == 0:
            recommendations.append({
                "icon": "🌐",
                "title": "Create a Website",
                "desc": "Having an online presence can increase sales by 15-25%",
                "priority": "High"
            })
        
        if marketing_binary == 0:
            recommendations.append({
                "icon": "📢", 
                "title": "Implement Marketing",
                "desc": "Marketing strategies boost visibility and customer acquisition",
                "priority": "High"
            })
        
        if rating < 4.0:
            recommendations.append({
                "icon": "⭐",
                "title": "Improve Customer Experience", 
                "desc": "Focus on service quality to increase ratings and repeat customers",
                "priority": "Medium"
            })
        
        if foot_traffic < 100:
            recommendations.append({
                "icon": "👥",
                "title": "Increase Foot Traffic",
                "desc": "Optimize location visibility or run promotional campaigns",
                "priority": "Medium"
            })
        
        if avg_order_value < 250:
            recommendations.append({
                "icon": "💰",
                "title": "Boost Order Value",
                "desc": "Implement upselling strategies and bundle offers",
                "priority": "Low"
            })
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
                priority_color = priority_colors[rec["priority"]]
                
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                            border-left: 4px solid {priority_color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.75rem;">{rec["icon"]}</span>
                        <h4 style="margin: 0; color: #1e293b; font-family: 'Poppins', sans-serif;">{rec["title"]}</h4>
                        <span style="background: {priority_color}; color: white; padding: 0.2rem 0.5rem; 
                                     border-radius: 6px; font-size: 0.75rem; margin-left: auto;">{rec["priority"]}</span>
                    </div>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem;">{rec["desc"]}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%); 
                        padding: 2rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎉</div>
                <h3 style="margin: 0; font-family: 'Poppins', sans-serif;">Outstanding Performance!</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                    Your shop has all the key success factors for optimal business performance!
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
