import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

# Page configuration
st.set_page_config(
    page_title="Food & Beverage Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        color: #495057;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #212529;
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: 700;
    }
    
    .info-card {
        background: linear-gradient(135deg, #e7f3ff 0%, #f0f8ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    .info-card h4 {
        color: #0066cc;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .section-header {
        color: #333;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.75rem;
        margin-bottom: 2rem;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 2.5rem;
        border-radius: 15px;
        border: 2px solid #b3d9ff;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .prediction-result h3 {
        color: #495057;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .prediction-result h1 {
        margin: 1rem 0;
        font-weight: 700;
        font-size: 2.8rem;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .insights-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('Dataset_for_Food_and_Beverages.csv')
        
        # Data cleaning
        df = df.sort_values("Shop_Name")
        df = df.drop("Shop_Id", axis=1)
        df = df.reset_index(drop=True)
        
        # Shop Type standardization
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
    
    regressor = RandomForestRegressor(random_state=42, n_estimators=100)
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
    
    classifier = RandomForestClassifier(random_state=42, n_estimators=100)
    classifier.fit(X_train_clf, y_train_clf)
    
    y_pred_clf = classifier.predict(X_test_clf)
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    
    return {
        'regressor': regressor,
        'classifier': classifier,
        'reg_metrics': {'mae': mae, 'r2': r2},
        'clf_metrics': {'accuracy': accuracy}
    }

def main():
    # Simple header
    st.title("Food & Beverage Analytics Dashboard")
    st.markdown("Business Intelligence and Prediction Platform")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Simple sidebar
    st.sidebar.header("Navigation")
    st.sidebar.markdown("Select a section to explore:")
    
    page = st.sidebar.selectbox(
        "Choose Section:",
        ["Dashboard", "Data Analysis", "Predictions"],
        help="Select different sections"
    )
    
    # Dataset info in sidebar
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.info(f"""
    **Total Records:** {len(df)}  
    **Average Rating:** {df['Rating'].mean():.1f}/5.0  
    **Data Quality:** Good
    """)
    
    # Page routing
    if page == "Dashboard":
        show_dashboard(df)
    elif page == "Data Analysis":
        show_data_analysis(df)
    elif page == "Predictions":
        show_predictions(df)

def show_dashboard(df):
    """Show main dashboard with key metrics and charts"""
    st.markdown('<h2 class="section-header">Business Overview Dashboard</h2>', unsafe_allow_html=True)
    
    # Enhanced Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Shops</h3>
            <h2>{len(df)}</h2>
            <p style="color: #28a745; margin: 0;">Active businesses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sales = df['Yearly_Sales'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Sales</h3>
            <h2>${avg_sales:,.0f}</h2>
            <p style="color: #28a745; margin: 0;">Per year</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_rating = df['Rating'].mean()
        rating_color = "#28a745" if avg_rating >= 4 else "#ffc107" if avg_rating >= 3 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Rating</h3>
            <h2>{avg_rating:.2f}</h2>
            <p style="color: {rating_color}; margin: 0;">Out of 5.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_traffic = df['Foot_Traffic'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Traffic</h3>
            <h2>{total_traffic:,}</h2>
            <p style="color: #28a745; margin: 0;">Monthly visits</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Shop Type Distribution")
        shop_dist = df['Shop_Type'].value_counts()
        
        # Enhanced pie chart with better colors
        colors = ['#0066cc', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14']
        fig = px.pie(values=shop_dist.values, names=shop_dist.index, title="")
        fig.update_traces(
            textposition='auto', 
            textinfo='percent+label',
            textfont_size=12,
            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2))
        )
        fig.update_layout(
            height=400, 
            showlegend=True,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Location Performance")
        location_sales = df.groupby('Shop_Location')['Yearly_Sales'].mean().sort_values(ascending=False)
        
        # Enhanced bar chart
        fig = px.bar(
            x=location_sales.index, 
            y=location_sales.values,
            title="", 
            labels={'x': 'Location', 'y': 'Average Sales ($)'},
            color=location_sales.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=400, 
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            showlegend=False
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Key Insights
    st.markdown('<h3 class="section-header">Key Business Insights</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_location = location_sales.index[0]
        st.markdown(f"""
        <div class="info-card">
            <h4>Top Performing Location</h4>
            <p><strong>{best_location}</strong></p>
            <p>${location_sales.iloc[0]:,.0f} average sales</p>
            <p style="color: #28a745; font-size: 0.9rem; margin: 0;">Leading market performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        marketing_impact = df.groupby('Marketing')['Yearly_Sales'].mean()
        impact_pct = ((marketing_impact.iloc[1] - marketing_impact.iloc[0]) / marketing_impact.iloc[0] * 100)
        st.markdown(f"""
        <div class="info-card">
            <h4>Marketing ROI Impact</h4>
            <p><strong>+{impact_pct:.1f}%</strong></p>
            <p>Sales increase with marketing</p>
            <p style="color: #0066cc; font-size: 0.9rem; margin: 0;">Strong positive correlation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_rated = len(df[df['Rating'] >= 4.0])
        high_rated_pct = (high_rated / len(df)) * 100
        st.markdown(f"""
        <div class="info-card">
            <h4>Customer Satisfaction</h4>
            <p><strong>{high_rated_pct:.1f}%</strong></p>
            <p>Shops with rating ≥ 4.0 stars</p>
            <p style="color: #28a745; font-size: 0.9rem; margin: 0;">Excellent service quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Performance Metrics
    st.markdown('<h3 class="section-header">Performance Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Sales vs Foot Traffic Correlation")
        fig = px.scatter(
            df, 
            x='Foot_Traffic', 
            y='Yearly_Sales',
            color='Shop_Type',
            size='Rating',
            hover_data=['Shop_Name', 'Average_Order_Value'],
            labels={'Foot_Traffic': 'Monthly Foot Traffic', 'Yearly_Sales': 'Annual Sales ($)'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Rating vs Sales Performance")
        fig = px.box(
            df, 
            x='Rating_Category', 
            y='Yearly_Sales',
            color='Rating_Category',
            labels={'Rating_Category': 'Rating Category', 'Yearly_Sales': 'Annual Sales ($)'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Dashboard Insights
    st.markdown('<h3 class="section-header">Market Intelligence</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Revenue vs Digital Presence")
        
        # Create digital presence categories
        df_digital = df.copy()
        df_digital['Digital_Presence'] = df_digital.apply(lambda x: 
            'Full Digital' if x['Shop_Website'] == 1 and x['Marketing'] == 1 
            else 'Partial Digital' if x['Shop_Website'] == 1 or x['Marketing'] == 1
            else 'No Digital', axis=1)
        
        digital_performance = df_digital.groupby('Digital_Presence')['Yearly_Sales'].agg(['mean', 'count']).reset_index()
        digital_performance.columns = ['Digital_Presence', 'Avg_Sales', 'Count']
        
        fig = px.bar(
            digital_performance,
            x='Digital_Presence',
            y='Avg_Sales',
            color='Avg_Sales',
            color_continuous_scale='Viridis',
            text='Count',
            labels={'Avg_Sales': 'Average Sales ($)', 'Digital_Presence': 'Digital Presence Level'}
        )
        fig.update_traces(texttemplate='%{text} shops', textposition='outside')
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Customer Satisfaction Distribution")
        
        rating_counts = df['Rating_Category'].value_counts()
        colors_rating = {'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'}
        
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            color=rating_counts.index,
            color_discrete_map=colors_rating,
            labels={'x': 'Rating Category', 'y': 'Number of Shops'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance Summary Cards
    st.markdown('<h3 class="section-header">Executive Summary</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_performer = df.loc[df['Yearly_Sales'].idxmax()]
        st.markdown(f"""
        <div class="info-card">
            <h4>Top Performer</h4>
            <p><strong>{top_performer['Shop_Name']}</strong></p>
            <p>${top_performer['Yearly_Sales']:,.0f} sales</p>
            <p style="color: #28a745; font-size: 0.9rem; margin: 0;">{top_performer['Rating']:.1f} ⭐ rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        website_impact = df.groupby('Shop_Website')['Yearly_Sales'].mean().diff().iloc[1]
        st.markdown(f"""
        <div class="info-card">
            <h4>Website ROI</h4>
            <p><strong>${website_impact:,.0f}</strong></p>
            <p>Additional revenue with website</p>
            <p style="color: #0066cc; font-size: 0.9rem; margin: 0;">Strong digital impact</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_order_impact = df['Average_Order_Value'].corr(df['Yearly_Sales'])
        st.markdown(f"""
        <div class="info-card">
            <h4>Order Value Impact</h4>
            <p><strong>{avg_order_impact:.3f}</strong></p>
            <p>Correlation with sales</p>
            <p style="color: #6f42c1; font-size: 0.9rem; margin: 0;">Moderate correlation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        growth_potential = len(df[(df['Shop_Website'] == 0) | (df['Marketing'] == 0)])
        st.markdown(f"""
        <div class="info-card">
            <h4>Growth Opportunity</h4>
            <p><strong>{growth_potential}</strong></p>
            <p>Shops with digital potential</p>
            <p style="color: #fd7e14; font-size: 0.9rem; margin: 0;">Untapped market</p>
        </div>
        """, unsafe_allow_html=True)

def show_predictions(df):
    """Show enhanced prediction interface"""
    st.markdown('<h2 class="section-header">Business Performance Predictions</h2>', unsafe_allow_html=True)
    
    # Train models
    with st.spinner("Training machine learning models..."):
        models = train_models(df)
    
    # Enhanced Model performance
    st.markdown('<h3 class="section-header">Model Performance Metrics</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        r2_score = models['reg_metrics']['r2']
        mae_score = models['reg_metrics']['mae']
        performance_color = "#28a745" if r2_score > 0.7 else "#ffc107" if r2_score > 0.5 else "#dc3545"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {performance_color};">
            <h3>Sales Prediction Model</h3>
            <h2 style="color: {performance_color};">{r2_score:.3f}</h2>
            <p><strong>R² Score</strong> - {r2_score*100:.1f}% variance explained</p>
            <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Mean Error: ${mae_score:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accuracy = models['clf_metrics']['accuracy']
        acc_color = "#28a745" if accuracy > 0.8 else "#ffc107" if accuracy > 0.6 else "#dc3545"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {acc_color};">
            <h3>Rating Prediction Model</h3>
            <h2 style="color: {acc_color};">{accuracy:.3f}</h2>
            <p><strong>Accuracy</strong> - {accuracy*100:.1f}% correct predictions</p>
            <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Classification success rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Prediction form
    st.markdown('<h3 class="section-header">Business Performance Predictor</h3>', unsafe_allow_html=True)
    st.markdown("Enter your shop characteristics to get AI-powered predictions:")
    
    # Form in a container
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Digital Presence & Marketing**")
            has_website = st.selectbox(
                "Website Status", 
                ["No", "Yes"],
                help="Does your business have a website?"
            )
            has_marketing = st.selectbox(
                "Marketing Activities", 
                ["No", "Yes"],
                help="Do you run marketing campaigns?"
            )
            rating = st.slider(
                "Current Customer Rating", 
                1.0, 5.0, 3.5, 0.1,
                help="Current average customer rating (1-5 stars)"
            )
        
        with col2:
            st.markdown("**Business Operations**")
            avg_order_value = st.number_input(
                "Average Order Value ($)", 
                min_value=0, 
                value=300,
                step=10,
                help="Average amount customers spend per visit"
            )
            foot_traffic = st.number_input(
                "Monthly Foot Traffic", 
                min_value=0, 
                value=100,
                step=5,
                help="Number of customers visiting monthly"
            )
    
    # Convert inputs
    website_binary = 1 if has_website == "Yes" else 0
    marketing_binary = 1 if has_marketing == "Yes" else 0
    
    # Centered prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "Generate AI Predictions", 
            help="Click to get sales and rating predictions",
            use_container_width=True,
            type="primary"
        )
    
    # Make predictions
    if predict_button:
        with st.spinner("Analyzing your business data with AI..."):
            # Sales prediction
            sales_input = np.array([[website_binary, marketing_binary, rating, avg_order_value, foot_traffic]])
            predicted_sales = models['regressor'].predict(sales_input)[0]
            
            # Rating category prediction
            rating_input = np.array([[website_binary, predicted_sales, avg_order_value, foot_traffic, marketing_binary]])
            predicted_rating_category = models['classifier'].predict(rating_input)[0]
        
        # Enhanced results display
        st.markdown('<h3 class="section-header">Prediction Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            sales_color = "#28a745" if predicted_sales > df['Yearly_Sales'].mean() else "#ffc107"
            st.markdown(f"""
            <div class="prediction-result" style="border-left: 4px solid {sales_color};">
                <h3>Predicted Annual Sales</h3>
                <h1 style="color: {sales_color};">${predicted_sales:,.0f}</h1>
                <p>Expected yearly revenue</p>
                <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                    {'Above' if predicted_sales > df['Yearly_Sales'].mean() else 'Below'} industry average
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color_map = {"Low": "#dc3545", "Medium": "#ffc107", "High": "#28a745"}
            color = color_map.get(predicted_rating_category, "#6c757d")
            
            st.markdown(f"""
            <div class="prediction-result" style="border-left: 4px solid {color};">
                <h3>Expected Rating Category</h3>
                <h1 style="color: {color};">{predicted_rating_category}</h1>
                <p>Customer satisfaction level</p>
                <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                    Predicted customer experience category
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Business comparison
        st.markdown('<h3 class="section-header">Market Comparison</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        industry_avg = df['Yearly_Sales'].mean()
        performance_vs_avg = ((predicted_sales - industry_avg) / industry_avg) * 100
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <h4>vs Industry Average</h4>
                <p><strong>{performance_vs_avg:+.1f}%</strong></p>
                <p>Performance difference</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            percentile = (df['Yearly_Sales'] < predicted_sales).mean() * 100
            st.markdown(f"""
            <div class="info-card">
                <h4>Market Percentile</h4>
                <p><strong>{percentile:.0f}th</strong></p>
                <p>Better than {percentile:.0f}% of shops</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            monthly_sales = predicted_sales / 12
            st.markdown(f"""
            <div class="info-card">
                <h4>Monthly Revenue</h4>
                <p><strong>${monthly_sales:,.0f}</strong></p>
                <p>Expected monthly sales</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced Recommendations
        st.markdown('<h3 class="section-header">Personalized Business Recommendations</h3>', unsafe_allow_html=True)
        
        recommendations = []
        
        if website_binary == 0:
            impact = df.groupby('Shop_Website')['Yearly_Sales'].mean().diff().iloc[1]
            recommendations.append({
                "title": "Create a Business Website",
                "description": f"Having a website could increase your sales by approximately ${impact:,.0f} per year",
                "priority": "High",
                "icon": "🌐"
            })
        
        if marketing_binary == 0:
            impact = df.groupby('Marketing')['Yearly_Sales'].mean().diff().iloc[1]
            recommendations.append({
                "title": "Implement Marketing Strategies",
                "description": f"Marketing activities could boost your sales by approximately ${impact:,.0f} annually",
                "priority": "High", 
                "icon": "📢"
            })
        
        if rating < 4.0:
            recommendations.append({
                "title": "Improve Customer Experience",
                "description": "Focus on service quality to increase ratings and customer retention",
                "priority": "Medium",
                "icon": "⭐"
            })
        
        if foot_traffic < df['Foot_Traffic'].median():
            recommendations.append({
                "title": "Increase Foot Traffic",
                "description": "Optimize location visibility, signage, or run promotional campaigns",
                "priority": "Medium",
                "icon": "👥"
            })
        
        if avg_order_value < df['Average_Order_Value'].median():
            recommendations.append({
                "title": "Boost Average Order Value",
                "description": "Implement upselling strategies, combo deals, or premium offerings",
                "priority": "Low",
                "icon": "💰"
            })
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_colors = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}
                priority_color = priority_colors[rec["priority"]]
                
                st.markdown(f"""
                <div class="info-card" style="border-left-color: {priority_color};">
                    <h4>{rec["icon"]} {rec["title"]} 
                        <span style="background: {priority_color}; color: white; padding: 0.2rem 0.6rem; 
                                     border-radius: 12px; font-size: 0.8rem; margin-left: 1rem;">
                            {rec["priority"]} Priority
                        </span>
                    </h4>
                    <p style="margin: 0.5rem 0 0 0;">{rec["description"]}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card" style="border-left-color: #28a745; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);">
                <h4>🎉 Excellent Business Setup!</h4>
                <p style="margin: 0.5rem 0 0 0;">
                    Your shop has all the key success factors for optimal performance. 
                    Continue maintaining these high standards!
                </p>
            </div>
            """, unsafe_allow_html=True)

def show_data_analysis(df):
    """Show detailed data analysis with advanced visualizations"""
    st.markdown('<h2 class="section-header">Advanced Data Analysis</h2>', unsafe_allow_html=True)
    
    # Interactive Filters
    st.markdown('<h3 class="section-header">Data Exploration Filters</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_types = st.multiselect(
            "Filter by Shop Type",
            options=df['Shop_Type'].unique(),
            default=df['Shop_Type'].unique()
        )
    
    with col2:
        selected_locations = st.multiselect(
            "Filter by Location", 
            options=df['Shop_Location'].unique(),
            default=df['Shop_Location'].unique()
        )
    
    with col3:
        rating_range = st.slider(
            "Rating Range",
            min_value=float(df['Rating'].min()),
            max_value=float(df['Rating'].max()),
            value=(float(df['Rating'].min()), float(df['Rating'].max())),
            step=0.1
        )
    
    # Filter data
    filtered_df = df[
        (df['Shop_Type'].isin(selected_types)) &
        (df['Shop_Location'].isin(selected_locations)) &
        (df['Rating'] >= rating_range[0]) &
        (df['Rating'] <= rating_range[1])
    ]
    
    # Display filtered data info
    st.info(f"Showing {len(filtered_df)} out of {len(df)} shops based on your filters")
    
    # Advanced Visualizations
    st.markdown('<h3 class="section-header">Business Intelligence Charts</h3>', unsafe_allow_html=True)
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Marketing vs Website Impact")
        marketing_website = filtered_df.groupby(['Marketing', 'Shop_Website'])['Yearly_Sales'].mean().reset_index()
        marketing_website['Marketing_Label'] = marketing_website['Marketing'].map({0: 'No Marketing', 1: 'Has Marketing'})
        marketing_website['Website_Label'] = marketing_website['Shop_Website'].map({0: 'No Website', 1: 'Has Website'})
        
        fig = px.bar(
            marketing_website,
            x='Marketing_Label',
            y='Yearly_Sales', 
            color='Website_Label',
            barmode='group',
            labels={'Yearly_Sales': 'Average Sales ($)', 'Marketing_Label': 'Marketing Status'},
            color_discrete_map={'No Website': '#dc3545', 'Has Website': '#28a745'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Average Order Value Distribution")
        fig = px.histogram(
            filtered_df,
            x='Average_Order_Value',
            nbins=20,
            color='Shop_Type',
            labels={'Average_Order_Value': 'Average Order Value ($)', 'count': 'Number of Shops'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Sales Performance Heatmap")
        sales_heatmap = filtered_df.pivot_table(
            values='Yearly_Sales',
            index='Shop_Type',
            columns='Shop_Location',
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(
            sales_heatmap,
            labels=dict(x="Location", y="Shop Type", color="Average Sales ($)"),
            aspect="auto",
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Rating vs Order Value")
        fig = px.scatter(
            filtered_df,
            x='Average_Order_Value',
            y='Rating',
            color='Shop_Location',
            size='Foot_Traffic',
            hover_data=['Shop_Name', 'Yearly_Sales'],
            labels={'Average_Order_Value': 'Average Order Value ($)', 'Rating': 'Customer Rating'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Correlation Analysis
    st.markdown('<h3 class="section-header">Correlation Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Feature Correlation Matrix")
        numeric_cols = ['Shop_Website', 'Marketing', 'Rating', 'Average_Order_Value', 'Foot_Traffic', 'Yearly_Sales']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            labels=dict(color="Correlation")
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Key Statistics")
        
        # Calculate key statistics
        correlation_sales_rating = filtered_df['Yearly_Sales'].corr(filtered_df['Rating'])
        correlation_traffic_sales = filtered_df['Foot_Traffic'].corr(filtered_df['Yearly_Sales'])
        correlation_order_sales = filtered_df['Average_Order_Value'].corr(filtered_df['Yearly_Sales'])
        
        st.markdown(f"""
        <div class="info-card">
            <h4>Correlation Insights</h4>
            <p><strong>Sales-Rating:</strong> {correlation_sales_rating:.3f}</p>
            <p><strong>Traffic-Sales:</strong> {correlation_traffic_sales:.3f}</p>
            <p><strong>Order Value-Sales:</strong> {correlation_order_sales:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Key Insights:**")
        if correlation_traffic_sales > 0.5:
            st.success("Strong traffic-sales relationship")
        if correlation_sales_rating > 0.3:
            st.info("Ratings impact sales positively")
        if correlation_order_sales > 0.4:
            st.info("Higher order values drive sales")
    
    # Data Table
    st.markdown('<h3 class="section-header">Filtered Data View</h3>', unsafe_allow_html=True)
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )
    
    # Additional Analysis Section
    st.markdown('<h3 class="section-header">Advanced Business Insights</h3>', unsafe_allow_html=True)
    
    # Performance metrics by location and type
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Revenue Distribution by Shop Type")
        fig = px.violin(
            filtered_df,
            x='Shop_Type',
            y='Yearly_Sales',
            color='Shop_Type',
            labels={'Shop_Type': 'Shop Type', 'Yearly_Sales': 'Annual Sales ($)'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            showlegend=False
        )
        fig.update_traces(meanline_visible=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Performance Metrics Comparison")
        
        # Create aggregated data for radar chart
        metrics_by_type = filtered_df.groupby('Shop_Type').agg({
            'Yearly_Sales': 'mean',
            'Rating': 'mean',
            'Foot_Traffic': 'mean',
            'Average_Order_Value': 'mean'
        }).round(2)
        
        # Normalize metrics for better visualization
        metrics_normalized = metrics_by_type.copy()
        for col in metrics_normalized.columns:
            metrics_normalized[col] = (metrics_normalized[col] - metrics_normalized[col].min()) / (metrics_normalized[col].max() - metrics_normalized[col].min()) * 100
        
        fig = px.line_polar(
            metrics_normalized.reset_index(),
            r='Yearly_Sales',
            theta=['Sales', 'Rating', 'Traffic', 'Order Value'],
            color='Shop_Type',
            line_close=True
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Trend Analysis
    st.markdown('<h3 class="section-header">Business Trends & Patterns</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Sales vs Rating Trend by Location")
        fig = px.scatter(
            filtered_df,
            x='Rating',
            y='Yearly_Sales',
            color='Shop_Location',
            size='Foot_Traffic',
            trendline="ols",
            hover_data=['Shop_Name', 'Shop_Type'],
            labels={'Rating': 'Customer Rating', 'Yearly_Sales': 'Annual Sales ($)'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Business Success Factors")
        
        # Create success score
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['Success_Score'] = (
            (filtered_df_copy['Yearly_Sales'] / filtered_df_copy['Yearly_Sales'].max()) * 0.4 +
            (filtered_df_copy['Rating'] / 5) * 0.3 +
            (filtered_df_copy['Foot_Traffic'] / filtered_df_copy['Foot_Traffic'].max()) * 0.3
        ) * 100
        
        success_by_features = pd.DataFrame({
            'Feature': ['Has Website', 'Has Marketing', 'High Rating (≥4)', 'High Traffic (≥median)'],
            'Success_Score': [
                filtered_df_copy[filtered_df_copy['Shop_Website'] == 1]['Success_Score'].mean(),
                filtered_df_copy[filtered_df_copy['Marketing'] == 1]['Success_Score'].mean(),
                filtered_df_copy[filtered_df_copy['Rating'] >= 4]['Success_Score'].mean(),
                filtered_df_copy[filtered_df_copy['Foot_Traffic'] >= filtered_df_copy['Foot_Traffic'].median()]['Success_Score'].mean()
            ]
        })
        
        fig = px.bar(
            success_by_features,
            x='Feature',
            y='Success_Score',
            color='Success_Score',
            color_continuous_scale='RdYlGn',
            labels={'Success_Score': 'Success Score (%)', 'Feature': 'Business Feature'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            showlegend=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
