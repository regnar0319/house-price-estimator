import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from src.translations import LANGUAGES, TRANSLATIONS

# --- Configuration & Data ---
st.set_page_config(page_title="Global Housing AI", page_icon="🌍", layout="wide")

# Currency Rates & Symbols
CURRENCIES = {
    'USD': 1.0, 'INR': 83.0, 'CNY': 7.2, 'EUR': 0.92, 'JPY': 150.0,
    'GBP': 0.79, 'BRL': 4.95, 'RUB': 91.0, 'TRY': 30.0, 'KRW': 1330.0,
    'SAR': 3.75
}
SYMBOLS = {
    'USD': '$', 'INR': '₹', 'CNY': '¥', 'EUR': '€', 'JPY': '¥',
    'GBP': '£', 'BRL': 'R$', 'RUB': '₽', 'TRY': '₺', 'KRW': '₩', 'SAR': '﷼'
}

# --- Helpers ---

# Translation
def get_txt(lang_code, key):
    lang_dict = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])
    # Fallback to English if key missing
    return lang_dict.get(key, TRANSLATIONS['en'].get(key, key))

# Reverse Geocoding
@st.cache_data(show_spinner=False)
def get_location_name(lat, lon, lang):
    try:
        geolocator = Nominatim(user_agent="global_housing_ai_v5")
        location = geolocator.reverse((lat, lon), language=lang, exactly_one=True)
        if location:
            return location.address
        return "Unknown Location"
    except Exception:
        return "Location Lookup Failed"

# --- Main App ---

@st.cache_resource
def load_model():
    return joblib.load('data/global_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading global model: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150?text=GlobalAI", width=120)
    st.title("⚙️ Settings")
    
    # Language Selector
    sel_lang_name = st.selectbox("Select Language", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[sel_lang_name]
    
    # Currency Selector
    sel_curr = st.radio(get_txt(lang_code, 'currency'), list(CURRENCIES.keys())[:5]) 
    rate = CURRENCIES.get(sel_curr, 1.0)
    symbol = SYMBOLS.get(sel_curr, '$')

# Header
st.title(get_txt(lang_code, 'title'))
st.caption(get_txt(lang_code, 'subtitle'))

# How it Works
with st.expander(f"ℹ️ {get_txt(lang_code, 'how_works')}"):
    st.markdown(get_txt(lang_code, 'how_works_text'))

st.markdown("---")

# Layout: Map & Form
col_map, col_form = st.columns([1.5, 1], gap="large")

# Init variables
lat, lon = None, None

with col_map:
    st.subheader(get_txt(lang_code, 'map_instr'))
    
    # Initialize Map - World View
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Output of map click
    map_output = st_folium(m, height=500, width=None)

    if map_output['last_clicked']:
        lat = map_output['last_clicked']['lat']
        lon = map_output['last_clicked']['lng']
        
        # Reverse Geocode
        loc_name = get_location_name(lat, lon, lang_code)
        
        st.success(f"**{get_txt(lang_code, 'lat_lon')}:** {lat:.4f}, {lon:.4f}")
        st.info(f"📍 **{loc_name}**")
        
    else:
        st.info(get_txt(lang_code, 'select_loc_warn'))

with col_form:
    st.subheader("📝 " + get_txt(lang_code, 'prop_details'))
    with st.form("main_form"):
        # Features: Latitude, Longitude, TotalArea, GarageCars, Bedrooms, HouseAge
        
        total_area = st.number_input(get_txt(lang_code, 'total_area'), min_value=300, max_value=10000, value=1500, step=50)
        
        # Columns for smaller inputs
        c1, c2 = st.columns(2)
        with c1:
            bedrooms = st.number_input(get_txt(lang_code, 'bedrooms'), min_value=1, max_value=10, value=3)
            garage_cars = st.selectbox(get_txt(lang_code, 'garage_cars'), [0, 1, 2, 3, 4], index=2)
        with c2:
            house_age = st.slider(get_txt(lang_code, 'house_age'), 0, 100, 10)
        
        submitted = st.form_submit_button(get_txt(lang_code, 'predict_btn'), use_container_width=True)

if submitted:
    if lat is None or lon is None:
        st.error(get_txt(lang_code, 'select_loc_warn'))
    else:
        # Prediction
        # Model expects: ['Latitude', 'Longitude', 'TotalArea', 'GarageCars', 'Bedrooms', 'HouseAge']
        input_df = pd.DataFrame({
            'Latitude': [lat],
            'Longitude': [lon],
            'TotalArea': [total_area],
            'GarageCars': [garage_cars],
            'Bedrooms': [bedrooms],
            'HouseAge': [house_age]
        })
        
        try:
            # Model returns value in $100,000s units
            pred_raw = model.predict(input_df)[0]
            pred_usd = pred_raw * 100000 
            
            # Apply currency conversion
            pred_final = pred_usd * rate
            
            st.balloons()
            st.markdown(f"### 💰 {get_txt(lang_code, 'result_title')}: {symbol}{pred_final:,.0f}")
                
        except Exception as e:
            st.error(str(e))

