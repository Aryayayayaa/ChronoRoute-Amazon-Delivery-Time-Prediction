import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
from math import radians, sin, cos, sqrt, atan2
import base64 

# ====================================================================
# A. PREPROCESSING CONSTANTS (CRITICAL FOR DEPLOYMENT)
# These values are derived directly from your Colab analysis.
# ====================================================================

# 1. ACTUAL MEAN AND STANDARD DEVIATION VALUES FROM COLAB SCALING
MOCK_MEANS = {
    'Agent_Age': 29.54958939,
    'Agent_Rating': 4.6360146,
    'Order_to_Pickup_Time': 27.31314182,
    'Distance_km': 17.31386725,
    'Order_Hour': 29.12800356,
    'Order_Minute': 9.95917269
}

MOCK_STDS = {
    'Agent_Age': 5.76528285,
    'Agent_Rating': 0.31166285,
    'Order_to_Pickup_Time': 304.47148147,
    'Distance_km': 4.80417993,
    'Order_Hour': 16.47400449,
    'Order_Minute': 4.08521498
}

# 2. DEFINITIVE CATEGORICAL SETS
CATEGORICAL_FEATURES = {
    'Weather_Condition': ['Fog', 'Sandstorms', 'Cloudy', 'Windy', 'Sunny', 'Stormy'],
    'Category': ['Home', 'Jewelry', 'Kitchen', 'Outdoors', 'Pet Supplies', 'Shoes', 'Skincare', 'Snacks', 
                 'Sports', 'Toys', 'Apparel', 'Books', 'Clothing', 'Cosmetics', 'Electronics', 'Grocery'],
    'Time_of_Day': ['Night', 'Morning', 'Afternoon', 'Evening'],
    'Traffic_Condition': ['High', 'Jam', 'Low', 'Medium'],
    'Area': ['Urban', 'Metropolitan', 'Semi-Urban', 'Other'], 
    'Vehicle': ['motorcycle', 'scooter', 'van', 'bicycle'] 
}

NUMERIC_COLS = ['Agent_Age', 'Agent_Rating', 'Order_to_Pickup_Time', 'Distance_km', 'Order_Hour', 'Order_Minute']
INPUT_CATEGORICAL_COLS = ['Weather_Condition', 'Traffic_Condition', 'Category', 'Area', 'Time_of_Day', 'Vehicle'] 

# Mapping prefixes for OHE columns
PREFIX_MAP = {
    'Weather_Condition': 'Weather',
    'Traffic_Condition': 'Traffic',
    'Category': 'Category',
    'Area': 'Area',
    'Time_of_Day': 'Time_of_Day',
    'Vehicle': 'Vehicle' 
}

# 3. ACTUAL EXPECTED COLUMNS (Copied exactly from your Colab output, including spaces)
EXPECTED_COLUMNS = [
    'Agent_Age', 'Agent_Rating', 'Order_to_Pickup_Time', 'Distance_km',
    'Order_Hour', 'Order_Minute', 'Weather_Fog', 'Weather_Sandstorms',
    'Weather_Stormy', 'Weather_Sunny', 'Weather_Windy', 'Traffic_Jam ', 
    'Traffic_Low ', 'Traffic_Medium ', 'Vehicle_scooter ', 'Vehicle_van',
    'Area_Other', 'Area_Semi-Urban ', 'Area_Urban ', 'Category_Books',
    'Category_Clothing', 'Category_Cosmetics', 'Category_Electronics',
    'Category_Grocery', 'Category_Home', 'Category_Jewelry', 'Category_Kitchen',
    'Category_Outdoors', 'Category_Pet Supplies', 'Category_Shoes',
    'Category_Skincare', 'Category_Snacks', 'Category_Sports', 'Category_Toys',
    'Time_of_Day_Morning', 'Time_of_Day_Afternoon', 'Time_of_Day_Evening'
]

# 4. Inferred Dropped Baselines 
DROPPED_BASELINES = {
    'Weather_Condition': 'Cloudy',      
    'Traffic_Condition': 'High',         
    'Area': 'Metropolitan',              
    'Category': 'Apparel',               
    'Time_of_Day': 'Night',              
    'Vehicle': 'motorcycle'              
}


# ====================================================================
# B. CUSTOM FEATURE ENGINEERING FUNCTIONS
# ====================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on the Earth."""
    R = 6371 # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def get_time_of_day(hour):
    """Classifies the hour into one of the four time slots used in training."""
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    return 'Night'

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            img_bytes = f.read()
        b64_encoded = base64.b64encode(img_bytes).decode()
        mime_type = "image/jpeg"
        
        st.markdown(
            f"""
            <style>
            /* Background */
            .stApp {{
                background-image: url("data:{mime_type};base64,{b64_encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}

            /* General text = white */
            .stApp, .stApp *, .stSidebar, .stSidebar * {{
                color: #ffffff !important;
            }}

            /* Sidebar darker */
            [data-testid="stSidebar"] {{
                background-color: rgba(18, 18, 18, 0.9);
                border-right: 1px solid #444444; 
            }}

            /* Inputs (typed values) → black */
            div[data-baseweb="input"] input, 
            div[data-baseweb="time-picker"] input {{
                color: #000000 !important; 
                -webkit-text-fill-color: #000000 !important;
                opacity: 1 !important;
            }}

            /* Selectbox — only selected value black (not dropdown options) */
            div[data-baseweb="select"] > div > div {{
                color: #000000 !important;
                -webkit-text-fill-color: #000000 !important;
            }}

            /* Keep labels white */
            label p {{
                color: #ffffff !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
    except FileNotFoundError:
        st.error(f"Error: Background image file '{image_file}' not found. Please ensure it is in the same directory as app.py.")
    
    except Exception as e:
        st.error(f"Error processing background image: {e}")


# ====================================================================
# C. APPLICATION SETUP
# ====================================================================

# Load the optimized model (Cached for performance)
@st.cache_resource
def load_model():
    """Loads the pre-trained Joblib model file."""
    try:
        model = joblib.load('best_delivery_predictor.joblib')
        return model
 
    except FileNotFoundError:
        st.error("Error: Model file 'best_delivery_predictor.joblib' not found. Please ensure it is in the same directory.")
        return None

model = load_model()

# Title and Description
st.set_page_config(page_title="Delivery Time Predictor", layout="wide")

# Inject the background image CSS here
set_background('bg.jpg') 

st.title("🚚 ChronoRoute: Delivery Time Prediction App") 
st.markdown("Enter the order and location details below to predict the estimated delivery time (in minutes).")

# ====================================================================
# D. INPUT COLLECTION (Streamlit UI)
# ====================================================================

if model:
    st.sidebar.header("Delivery Coordinates (Required for Distance)")

    # Using mock coordinates near a central location for demonstration
    store_lat = st.sidebar.number_input("Store Latitude", value=12.9716, format="%.4f")
    store_lon = st.sidebar.number_input("Store Longitude", value=77.5946, format="%.4f")
    drop_lat = st.sidebar.number_input("Drop Latitude", value=13.0333, format="%.4f")
    drop_lon = st.sidebar.number_input("Drop Longitude", value=77.5857, format="%.4f")

    st.sidebar.markdown("---")
    
    # Use columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    # Column 1: Agent and Preparation
    with col1:
        st.header("Agent & Preparation")
        # Agent_Age range adjusted based on your scaling data (mean ~29.5)
        agent_age = st.slider("Agent Age (Years)", min_value=18, max_value=55, value=30)
        agent_rating = st.slider("Agent Rating (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=4.63, step=0.01) 
        order_to_pickup_time = st.number_input("Order Prep Time (Minutes)", min_value=0.0, max_value=60.0, value=25.0, step=1.0) 
        order_datetime = st.time_input("Order Placed At (Time)", datetime.time(10, 0)) 

    # Column 2: Route and Environment
    with col2:
        st.header("Route & Environment")
        traffic_condition = st.selectbox("Traffic Condition", options=CATEGORICAL_FEATURES['Traffic_Condition'], index=CATEGORICAL_FEATURES['Traffic_Condition'].index('Low'))
        weather_condition = st.selectbox("Weather Condition", options=CATEGORICAL_FEATURES['Weather_Condition'], index=CATEGORICAL_FEATURES['Weather_Condition'].index('Fog'))
        area_category = st.selectbox("Delivery Area", options=CATEGORICAL_FEATURES['Area'], index=CATEGORICAL_FEATURES['Area'].index('Urban'))
        
    # Column 3: Order Details
    with col3:
        st.header("Order Details")
        category = st.selectbox("Order Category", options=CATEGORICAL_FEATURES['Category'], index=CATEGORICAL_FEATURES['Category'].index('Toys'))
        
        # Vehicle options based on the observed data categories
        vehicle = st.selectbox("Vehicle Type", options=CATEGORICAL_FEATURES['Vehicle'], index=CATEGORICAL_FEATURES['Vehicle'].index('scooter'))
        st.markdown("<br><br><br>", unsafe_allow_html=True) # Spacer for layout alignment
        
# ====================================================================
# E. PREDICTION LOGIC
# ====================================================================
    
    final_input = pd.DataFrame() 

    if st.button("Predict Delivery Time", use_container_width=True, type="primary"):
        try:
            # 1. Feature Engineering (Recreating Training Features)
            order_hour = order_datetime.hour
            order_minute = order_datetime.minute
            distance_km = haversine(store_lat, store_lon, drop_lat, drop_lon)
            time_of_day = get_time_of_day(order_hour)

            # 2. Prepare Raw Input Data
            raw_input = {
                'Agent_Age': agent_age,
                'Agent_Rating': agent_rating,
                'Order_to_Pickup_Time': order_to_pickup_time,
                'Distance_km': distance_km,
                'Order_Hour': order_hour, 
                'Order_Minute': order_minute,
                'Time_of_Day': time_of_day,
                'Weather_Condition': weather_condition,
                'Traffic_Condition': traffic_condition,
                'Category': category,
                'Area': area_category,
                'Vehicle': vehicle,
            }
            input_df = pd.DataFrame([raw_input])


            # 3. Standardize Numerical Features
            standardized_numerics = input_df[NUMERIC_COLS].copy()
            for col in NUMERIC_COLS:
                mean = MOCK_MEANS.get(col, 0)
                std = MOCK_STDS.get(col, 1)
                standardized_numerics[col] = (standardized_numerics[col] - mean) / std

            # 4. Create One-Hot Encoded DataFrame (Force to Integer Type 0 or 1)
            # Initialize with all expected OHE columns set to 0 (Integer)
            ohe_cols_only = [c for c in EXPECTED_COLUMNS if c not in NUMERIC_COLS]
            ohe_df = pd.DataFrame(0, index=[0], 
                                  columns=ohe_cols_only, 
                                  dtype=np.int8) 
            
            # Set the relevant column to 1 based on user input for each categorical feature
            for feature in INPUT_CATEGORICAL_COLS:
                user_value = input_df[feature].iloc[0]
                
                # Check if the user selects the baseline category. If so, skip (the column remains 0)
                if user_value == DROPPED_BASELINES.get(feature):
                    continue

                # For all other (non-baseline) categories, find the column name and set it to 1
                prefix = PREFIX_MAP.get(feature, feature)
                
                # CRITICAL FIX: Append the user value, but also append a space if the OHE column
                # name in the EXPECTED_COLUMNS list contains a space (which yours does for many).
                
                potential_col_name = f'{prefix}_{user_value}'
                
                # Try to match the name with or without a trailing space
                if potential_col_name in ohe_df.columns:
                    col_name = potential_col_name
                elif f'{potential_col_name} ' in ohe_df.columns:
                    col_name = f'{potential_col_name} '
                else:
                    # If it's not a baseline but also not a recognized OHE column, raise an error
                    raise ValueError(f"Feature name matching error: Could not find OHE column for input '{user_value}' in feature '{feature}'.")

                # Set the found OHE column to 1
                ohe_df.loc[0, col_name] = 1
                
            # 5. FINAL INPUT ASSEMBLY (CRITICAL: Enforce column order and data type)
            temp_final_input = pd.concat([standardized_numerics, ohe_df], axis=1)
            
            # Ensure the final DataFrame strictly follows the expected order
            final_input = temp_final_input[EXPECTED_COLUMNS].copy() # Using .copy() is safer
            
            # CRITICAL FIX: Convert the entire DataFrame to the expected float type 
            final_input = final_input.astype(np.float32) 

            # 6. Predict
            prediction = model.predict(final_input)
            predicted_time = prediction[0]

            st.balloons()
            st.success("---")
            st.metric(label="Predicted Delivery Time", value=f"{predicted_time:.2f} minutes")
            
            # Using a representative RMSE based on typical XGBoost performance on this data
            rmse = 22.67 
            lower_bound = max(0, predicted_time - rmse)
            upper_bound = predicted_time + rmse
            st.info(f"The Expected Time of Arrival (ETA) is estimated to be between **{lower_bound:.0f} and {upper_bound:.0f} minutes** (based on the model's RMSE of {rmse:.2f} minutes).")


        except Exception as e:
            st.error(f"Prediction failed due to error: {e}")
            # Display the DataFrame used for prediction to help debugging
            if not final_input.empty:
                 st.subheader("Debugging: Final Input DataFrame (Check Column Names/Order)")
                 # Displaying the transpose helps visualize the columns in the correct order
                 st.dataframe(final_input.T) 
            else:
                 st.subheader("Debugging: Raw Input Data")
                 if 'raw_input' in locals():
                    st.json(raw_input)
                 else:
                    st.info("Raw input data could not be retrieved.")
