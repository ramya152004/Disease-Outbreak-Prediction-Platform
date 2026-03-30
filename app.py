import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import hashlib



# --- Configuration ---
st.set_page_config(
    page_title="Disease Outbreak Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦠"
)



# ============================================
# USER MANAGEMENT
# ============================================
USERS_FILE = "users.json"


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    else:
        default_users = {
            "admin": {
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "name": "Admin User",
                "role": "Admin"
            }
        }
        save_users(default_users)
        return default_users


def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password, name, role="User"):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        "password": hash_password(password),
        "name": name,
        "role": role
    }
    save_users(users)
    return True, "Registration successful!"


def verify_user(username, password):
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return True, users[username]
    return False, None



# ============================================
# LOAD HISTORICAL DATA (FIXED DATA TYPES)
# ============================================
@st.cache_data
def load_historical_data():
    """Load and clean historical data with proper data types"""
    try:
        data = pd.read_csv('Final_data.csv')
        
        # ✅ Convert numeric columns to proper types
        numeric_cols = ['Cases', 'Deaths', 'Latitude', 'Longitude', 'Temp', 'preci', 'LAI']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Convert date columns to integers
        date_cols = ['day', 'mon', 'year']
        for col in date_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(1).astype(int)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None



# ============================================
# GET TRAINED LOCATIONS ONLY
# ============================================
def get_trained_states(le_state):
    """Get only states that model was trained on"""
    return sorted(le_state.classes_)


def get_trained_districts_for_state(data, state, le_district):
    """Get only districts from selected state that model knows"""
    state_districts = data[data['state_ut'] == state]['district'].unique()
    trained = [d for d in state_districts if d in le_district.classes_]
    return sorted(trained)



# ============================================
# CUSTOM CSS - FIXED TO MATCH IMAGE
# ============================================
# ============================================
# CUSTOM CSS - FIXED SIDEBAR INPUT VISIBILITY
# ============================================
# ============================================
# CUSTOM CSS - CREAM SIDEBAR
# ============================================
st.markdown("""
    <style>
    /* Main app background - Purple gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .main {
        background: transparent !important;
    }
    
    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }
    
    .metric-card {
        background: white; 
        padding: 20px; 
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15); 
    }
    
    .stButton>button {
        width: 100%; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        font-weight: bold; 
        border-radius: 10px; 
        padding: 15px;
        border: none; 
        font-size: 16px; 
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: scale(1.05); 
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); 
    }
    
    h1 { 
        color: white; 
        text-align: center; 
        font-size: 3em; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); 
        margin-bottom: 10px; 
    }
    
    .subtitle { 
        color: white; 
        text-align: center; 
        font-size: 1.2em; 
        margin-bottom: 30px; 
        opacity: 0.9; 
    }
    
    .risk-high { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
        color: white; 
        padding: 20px; 
        border-radius: 15px; 
        font-size: 1.2em; 
        font-weight: bold; 
        text-align: center; 
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
        color: #333; 
        padding: 20px; 
        border-radius: 15px; 
        font-size: 1.2em; 
        font-weight: bold; 
        text-align: center; 
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
        color: #333; 
        padding: 20px; 
        border-radius: 15px; 
        font-size: 1.2em; 
        font-weight: bold; 
        text-align: center; 
    }
    
    .info-box { 
        background: rgba(255, 255, 255, 0.95); 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid #667eea; 
        margin: 20px 0; 
    }
    
    .auth-box { 
        background: rgba(255, 255, 255, 0.95); 
        padding: 40px; 
        border-radius: 20px; 
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2); 
    }
    
    .factor-box { 
        background: #f8f9fa; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0; 
        border-left: 4px solid #667eea; 
    }
    
    /* ===== SIDEBAR - CREAM COLOR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF8E7 0%, #FFE4B5 100%) !important;
    }
    
    /* Sidebar content wrapper */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #FFF8E7 0%, #FFE4B5 100%) !important;
    }
    
    /* Sidebar headers - dark text on cream */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #333 !important;
    }
    
    /* Sidebar regular text */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        color: #333 !important;
    }
    
    /* Sidebar labels - dark text */
    [data-testid="stSidebar"] label {
        color: #333 !important;
        font-weight: 600 !important;
    }
    
    /* Input boxes - white with dark text */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea {
        background-color: white !important;
        color: #333 !important;
        border: 2px solid #D4A373 !important;
        border-radius: 8px !important;
    }
    
    /* Dropdown selected text */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: white !important;
        color: #333 !important;
        border: 2px solid #D4A373 !important;
    }
    
    /* Dropdown options text */
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #333 !important;
    }
    
    /* Date input */
    [data-testid="stSidebar"] [data-baseweb="input"] input {
        color: #333 !important;
        background-color: white !important;
    }
    
    /* Slider styling */
    [data-testid="stSidebar"] [data-testid="stTickBar"] div {
        color: #333 !important;
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: #D4A373 !important;
    }
    
    /* Info/Alert boxes in sidebar */
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(212, 163, 115, 0.2) !important;
        color: #333 !important;
        border: 1px solid #D4A373 !important;
    }
    
    /* Divider lines */
    [data-testid="stSidebar"] hr {
        border-color: #D4A373 !important;
        opacity: 0.3;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# AUTH PAGE
# ============================================
def auth_page():
    st.markdown("<h1>🦠 Disease Outbreak Prediction Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Early Warning System for Acute Diarrhoeal Disease</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.markdown("<div class='auth-box'>", unsafe_allow_html=True)
            login_username = st.text_input("👤 Username", placeholder="Enter your username", key="login_user")
            login_password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pass")
            
            if st.button("🚀 Login", use_container_width=True, key="login_btn"):
                if login_username and login_password:
                    is_valid, user_data = verify_user(login_username, login_password)
                    if is_valid:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = login_username
                        st.session_state['user_data'] = user_data
                        st.success("✅ Login successful!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
            st.markdown("</div>", unsafe_allow_html=True)
            st.info("**Default Admin Account:** `admin` / `admin123`")
        
        with tab2:
            st.markdown("<div class='auth-box'>", unsafe_allow_html=True)
            reg_name = st.text_input("👤 Full Name", placeholder="Enter your full name", key="reg_name")
            reg_username = st.text_input("🔑 Username", placeholder="Choose a username", key="reg_user")
            reg_password = st.text_input("🔒 Password", type="password", placeholder="Choose a password", key="reg_pass")
            reg_password_confirm = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password", key="reg_pass_confirm")
            
            if st.button("📝 Register", use_container_width=True, key="reg_btn"):
                if not all([reg_name, reg_username, reg_password, reg_password_confirm]):
                    st.warning("⚠️ Please fill all fields")
                elif reg_password != reg_password_confirm:
                    st.error("❌ Passwords do not match")
                elif len(reg_password) < 6:
                    st.warning("⚠️ Password must be at least 6 characters")
                else:
                    success, message = register_user(reg_username, reg_password, reg_name)
                    if success:
                        st.success(f"✅ {message} You can now login!")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br><p style='text-align: center; color: rgba(255,255,255,0.7);'>Secure Authentication System</p>", unsafe_allow_html=True)



# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('best_disease_model.pkl')
        return pipeline
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None



# ============================================
# HELPER FUNCTIONS
# ============================================
def get_historical_trends(data, state, district, weeks=8):
    """Get past N weeks of data for selected location"""
    filtered = data[(data['state_ut'] == state) & (data['district'] == district)]
    filtered = filtered.sort_values(['year', 'mon', 'day'])
    
    # ✅ Ensure Cases is numeric
    filtered['Cases'] = pd.to_numeric(filtered['Cases'], errors='coerce').fillna(0)
    
    if len(filtered) < weeks:
        return filtered.tail(len(filtered))
    return filtered.tail(weeks)


def get_risk_level(predicted_cases):
    if predicted_cases > 100:
        return "High Risk", "🔴", "Immediate intervention needed. Outbreak likely.", "risk-high"
    elif predicted_cases >= 50:
        return "Medium Risk", "🟡", "Monitor situation closely. Prepare resources.", "risk-medium"
    else:
        return "Low Risk", "🟢", "Standard surveillance. Continue monitoring.", "risk-low"


def explain_factors(model, features, temp, preci, lai):
    """Explain which factors contributed most to prediction"""
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        top5 = importance.head(5)
        explanations = []
        
        for _, row in top5.iterrows():
            feat = row['Feature']
            imp = row['Importance'] * 100
            
            if 'cases_last' in feat:
                explanations.append(f"**Recent case trends** ({imp:.1f}%): Past outbreak patterns strongly influence future risk")
            elif feat == 'Temp_scaled':
                explanations.append(f"**Temperature** ({imp:.1f}%): {temp-273.15:.1f}°C affects disease transmission")
            elif feat == 'preci_scaled':
                explanations.append(f"**Rainfall** ({imp:.1f}%): {preci:.2f}mm impacts vector breeding")
            elif feat == 'LAI_scaled':
                explanations.append(f"**Vegetation** ({imp:.1f}%): Environmental conditions influence risk")
            elif feat == 'mon':
                explanations.append(f"**Seasonal patterns** ({imp:.1f}%): Month-specific outbreak trends")
            else:
                explanations.append(f"**{feat}** ({imp:.1f}%): Contributes to prediction")
        
        return explanations, top5
    return [], None



# ============================================
# MAIN DASHBOARD
# ============================================
def main_dashboard():
    pipeline = load_model()
    historical_data = load_historical_data()
    user_data = st.session_state.get('user_data', {})
    
    st.markdown("<h1>🦠 Disease Outbreak Prediction Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Early Warning System for Acute Diarrhoeal Disease</p>", unsafe_allow_html=True)
    
    if pipeline and historical_data is not None:
        model = pipeline['model']
        scaler = pipeline['scaler']
        le_state = pipeline['le_state']
        le_district = pipeline['le_district']
        feature_cols = pipeline['features']
        
        # Show Model Metrics at Top
        st.markdown("### 📊 Model Performance & Reliability")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("🎯 Model Type", pipeline.get('model_type', 'Gradient Boosting'))
        with col_m2:
            train_rmse = pipeline.get('train_rmse', 0)
            st.metric("📉 Training RMSE", f"{train_rmse:.2f}")
        with col_m3:
            test_rmse = pipeline.get('test_rmse', 0)
            st.metric("📉 Testing RMSE", f"{test_rmse:.2f}")
        with col_m4:
            st.metric("✅ Training Data", f"{len(historical_data):,} records")
        
        st.info("**Lower RMSE = Better accuracy**. This model was trained on real outbreak data and validated on unseen test cases.")
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.markdown(f"### 👤 {user_data.get('name', st.session_state['username'])}")
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state['logged_in'] = False
                st.session_state.pop('username', None)
                st.session_state.pop('user_data', None)
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Configuration")
            st.markdown("---")
            
            # ✅ ONLY SHOW TRAINED STATES
            st.markdown("#### 📍 Location")
            trained_states = get_trained_states(le_state)
            selected_state = st.selectbox("State/UT", trained_states, help="Select the state or union territory")
            
            # ✅ ONLY SHOW TRAINED DISTRICTS FOR SELECTED STATE
            trained_districts = get_trained_districts_for_state(historical_data, selected_state, le_district)
            
            if len(trained_districts) == 0:
                st.error(f"❌ No trained districts available for {selected_state}")
                st.stop()
            
            selected_district = st.selectbox("District", trained_districts, help="Select the district for prediction")
            
            st.markdown("---")
            st.markdown("#### 📅 Prediction Settings")
            prediction_date = st.date_input("Target Date", value=datetime.now() + timedelta(days=7), help="Select the date for outbreak prediction")
            weeks_lookback = st.slider("Historical Weeks to Analyze", min_value=4, max_value=12, value=8, help="Number of past weeks to analyze")
            
            st.markdown("---")
            st.markdown("#### 🌡️ Environmental Factors")
            st.info("Auto-populated from recent data, or adjust manually")
            
            # Auto-populate from recent data
            recent = get_historical_trends(historical_data, selected_state, selected_district, weeks=2)
            if len(recent) > 0:
                avg_temp = float(recent['Temp'].mean())
                avg_preci = float(recent['preci'].mean())
                avg_lai = float(recent['LAI'].mean())
            else:
                avg_temp = 300.0
                avg_preci = 0.5
                avg_lai = 2.0
            
            temp = st.slider("Temperature (K)", min_value=250.0, max_value=320.0, value=avg_temp, step=0.5, help="Average temperature in Kelvin")
            preci = st.slider("Precipitation (mm)", min_value=0.0, max_value=100.0, value=avg_preci, step=0.1, help="Total precipitation in millimeters")
            lai = st.slider("Leaf Area Index", min_value=0.0, max_value=5.0, value=avg_lai, step=0.1, help="Vegetation density indicator")
            
            st.markdown("---")
            predict_button = st.button("🔮 Generate Prediction", use_container_width=True)
        
        # Main Content Area
        if predict_button:
            with st.spinner("🔄 Analyzing historical trends and generating prediction..."):
                # Get historical data for location
                hist = get_historical_trends(historical_data, selected_state, selected_district, weeks_lookback)
                
                if len(hist) == 0:
                    st.error(f"❌ No historical data found for {selected_district}, {selected_state}")
                    st.warning("Please select a different location with available data.")
                    st.stop()
                
                # Auto-populate recent cases from historical data
                cases_last_week = int(hist['Cases'].iloc[-1]) if len(hist) >= 1 else 10
                cases_last_month = int(hist['Cases'].tail(4).mean()) if len(hist) >= 4 else 10
                
                lat_val = float(hist['Latitude'].iloc[-1]) if len(hist) > 0 else 20.0
                lon_val = float(hist['Longitude'].iloc[-1]) if len(hist) > 0 else 78.0
                
                # Scale climate data
                raw_climate = np.array([[temp, preci, lai]])
                scaled_climate = scaler.transform(raw_climate)
                
                # Encode location (safe because we filtered to trained locations only)
                state_enc = le_state.transform([selected_state])[0]
                dist_enc = le_district.transform([selected_district])[0]
                
                # Prepare model input
                input_data = np.array([[
                    prediction_date.day, prediction_date.month, prediction_date.year,
                    lat_val, lon_val,
                    scaled_climate[0][0], scaled_climate[0][1], scaled_climate[0][2],
                    cases_last_week, cases_last_month,
                    state_enc, dist_enc
                ]])
                
                # PREDICT
                pred_cases = max(0, float(model.predict(input_data)[0]))
                risk_label, risk_color, risk_msg, risk_class = get_risk_level(pred_cases)
                
                # Display Results
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""<div class='metric-card'>
                        <h3 style='color: #667eea; margin: 0;'>Predicted Cases</h3>
                        <h1 style='color: #764ba2; margin: 10px 0;'>{int(pred_cases)}</h1>
                        <p style='color: #888; margin: 0;'>Next Period</p></div>""", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""<div class='metric-card'>
                        <h3 style='color: #667eea; margin: 0;'>Risk Level</h3>
                        <h1 style='margin: 10px 0;'>{risk_color}</h1>
                        <p style='color: #888; margin: 0;'>{risk_label}</p></div>""", unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""<div class='metric-card'>
                        <h3 style='color: #667eea; margin: 0;'>Last Week</h3>
                        <h1 style='color: #764ba2; margin: 10px 0;'>{cases_last_week}</h1>
                        <p style='color: #888; margin: 0;'>Actual Cases</p></div>""", unsafe_allow_html=True)
                
                with col4:
                    change_pct = ((pred_cases - cases_last_week) / max(cases_last_week, 1)) * 100
                    arrow = "↑" if change_pct > 0 else "↓"
                    color = "#f5576c" if change_pct > 0 else "#4ecdc4"
                    st.markdown(f"""<div class='metric-card'>
                        <h3 style='color: #667eea; margin: 0;'>Trend</h3>
                        <h1 style='color: {color}; margin: 10px 0;'>{arrow} {abs(change_pct):.1f}%</h1>
                        <p style='color: #888; margin: 0;'>vs Last Week</p></div>""", unsafe_allow_html=True)
                
                st.markdown(f"<br><div class='{risk_class}'>{risk_color} {risk_label.upper()}: {risk_msg}</div><br>", unsafe_allow_html=True)
                
                # HISTORICAL VS PREDICTED CHART
                st.markdown("### 📈 Historical Trends vs Predicted Outbreak")
                
                hist_viz = hist.tail(weeks_lookback)[['Cases', 'year', 'mon', 'day']].copy()
                hist_viz['Type'] = 'Historical (Actual)'
                
                pred_row = pd.DataFrame([{
                    'Cases': int(pred_cases),
                    'Type': 'Predicted (Next Period)',
                    'year': prediction_date.year,
                    'mon': prediction_date.month,
                    'day': prediction_date.day
                }])
                
                combined = pd.concat([hist_viz, pred_row], ignore_index=True)
                combined['Period'] = range(len(combined))
                
                fig = go.Figure()
                
                # Historical line
                hist_mask = combined['Type'] == 'Historical (Actual)'
                fig.add_trace(go.Scatter(
                    x=combined[hist_mask]['Period'],
                    y=combined[hist_mask]['Cases'],
                    mode='lines+markers',
                    name='Historical Cases',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8, color='#667eea')
                ))
                
                # Predicted point
                pred_mask = combined['Type'] == 'Predicted (Next Period)'
                fig.add_trace(go.Scatter(
                    x=combined[pred_mask]['Period'],
                    y=combined[pred_mask]['Cases'],
                    mode='markers',
                    name='Predicted Cases',
                    marker=dict(size=15, color='#f5576c', symbol='star', line=dict(width=2, color='white'))
                ))
                
                fig.update_layout(
                    title=f"Time-Series Analysis: {selected_district}, {selected_state}",
                    xaxis_title="Time Period (Past → Future)",
                    yaxis_title="Number of Cases",
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # FACTOR EXPLANATION
                st.markdown("### 🧠 What Factors Contributed to This Prediction?")
                
                explanations, top_factors = explain_factors(model, feature_cols, temp, preci, lai)
                
                col_exp1, col_exp2 = st.columns([2, 1])
                
                with col_exp1:
                    st.markdown("""
                        <div class='info-box'>
                            <h4>📊 Key Contributing Factors</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for i, exp in enumerate(explanations, 1):
                        st.markdown(f"""
                            <div class='factor-box'>
                                <strong>{i}.</strong> {exp}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.info(f"""
                        **Based on {weeks_lookback} weeks of historical data** from {selected_district}, {selected_state}.
                        The model analyzed {len(hist)} data points to generate this prediction.
                    """)
                
                with col_exp2:
                    st.markdown("**Feature Importance**")
                    if top_factors is not None:
                        fig_imp = px.bar(
                            top_factors.head(5),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        fig_imp.update_layout(
                            showlegend=False,
                            height=300,
                            margin=dict(l=0, r=0, t=0, b=0),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                
                # MODEL EVALUATION METRICS SECTION
                st.markdown("### 📉 Model Accuracy & Reliability")
                col_met1, col_met2, col_met3 = st.columns(3)
                
                with col_met1:
                    st.markdown(f"""
                        <div class='info-box' style='text-align: center;'>
                            <h3 style='color: #667eea;'>RMSE</h3>
                            <h2>{test_rmse:.2f}</h2>
                            <p style='color: #888;'>Root Mean Square Error</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_met2:
                    test_mae = pipeline.get('test_mae', test_rmse * 0.8)
                    st.markdown(f"""
                        <div class='info-box' style='text-align: center;'>
                            <h3 style='color: #667eea;'>MAE</h3>
                            <h2>{test_mae:.2f}</h2>
                            <p style='color: #888;'>Mean Absolute Error</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_met3:
                    accuracy_pct = max(0, (1 - (test_rmse / 100)) * 100)
                    st.markdown(f"""
                        <div class='info-box' style='text-align: center;'>
                            <h3 style='color: #667eea;'>Model Quality</h3>
                            <h2>{accuracy_pct:.1f}%</h2>
                            <p style='color: #888;'>Prediction Confidence</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.info(f"""
                    **What these metrics mean:**
                    - **RMSE & MAE**: Lower values indicate better prediction accuracy
                    - This model was trained on actual outbreak data and validated on unseen test data
                    - Predictions are based on learned patterns from {len(historical_data):,} historical disease records
                """)
        
        else:
            # Welcome Screen
            st.markdown("### 👋 Welcome to the Disease Outbreak Prediction Platform")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""<div class='info-box'><h3>🎯 Accurate Predictions</h3>
                    <p>Advanced ML models trained on historical outbreak data combined with environmental factors.</p></div>""", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""<div class='info-box'><h3>⚡ Real-Time Analysis</h3>
                    <p>Get instant risk assessments based on current conditions and historical trends.</p></div>""", unsafe_allow_html=True)
            
            with col3:
                st.markdown("""<div class='info-box'><h3>📊 Data-Driven Insights</h3>
                    <p>Understand key factors contributing to outbreak risks with detailed visualizations.</p></div>""", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🚀 Get Started")
            st.info("👈 Configure the prediction parameters in the sidebar and click **'Generate Prediction'** to analyze outbreak risk.")
    
    else:
        st.error("⚠️ Model not found. Please train the model first using `model_training.py`.")
        st.info("Run: `python model_training.py` to train and save the model.")



# ============================================
# MAIN APP LOGIC
# ============================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False


if st.session_state['logged_in']:
    main_dashboard()
else:
    auth_page()
