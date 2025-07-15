import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

# Add src to path for importing preprocessing
sys.path.append('./src')
from preprocess import preprocess_data

# Set page config
st.set_page_config(
    page_title="YourCabs Cancellation Predictor",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model and metadata"""
    try:
        # Load model
        model = joblib.load('./models/best_model_xgboost.joblib')
        
        # Load metadata
        with open('./models/model_metadata_xgboost.json', 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def main():
    st.title("🚖 YourCabs Cancellation Predictor")
    st.markdown("---")
    
    # Load model
    model, metadata = load_model()
    
    if model is None:
        st.error("Could not load the model. Please check if model files exist.")
        return
    
    # Display model info
    with st.expander("ℹ️ Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", metadata['model_name'])
        with col2:
            st.metric("AUC Score", f"{metadata['auc_score']:.3f}")
        with col3:
            st.metric("Accuracy", f"{metadata['accuracy']:.3f}")
    
    st.markdown("## 📊 Predict Cab Cancellation")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Booking Details")
            online_booking = st.selectbox("Online Booking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            mobile_site_booking = st.selectbox("Mobile Site Booking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            booking_hour = st.slider("Booking Hour", 0, 23, 12)
            is_weekend_booking = st.selectbox("Weekend Booking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            
            st.subheader("Travel Details")
            travel_type = st.selectbox("Travel Type", [1, 2, 3], format_func=lambda x: f"Type {x}")
            is_business_travel = st.selectbox("Business Travel", [0, 1], format_func=lambda x: "Yes" if x else "No")
            is_leisure_travel = st.selectbox("Leisure Travel", [0, 1], format_func=lambda x: "Yes" if x else "No")
            trip_distance = st.number_input("Trip Distance (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        
        with col2:
            st.subheader("Booking Patterns")
            business_advance_booking = st.selectbox("Business Advance Booking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            leisure_last_minute = st.selectbox("Leisure Last Minute", [0, 1], format_func=lambda x: "Yes" if x else "No")
            
            st.subheader("Vehicle & Location")
            is_model_12 = st.selectbox("Model 12 Vehicle", [0, 1], format_func=lambda x: "Yes" if x else "No")
            from_city_id = st.selectbox("From City ID", [5.0, 15.0, 25.0, 35.0], format_func=lambda x: f"City {x}")
            
            st.markdown("### Additional Features")
            st.caption("Most other features will be auto-calculated based on your inputs above")
        
        submitted = st.form_submit_button("🔮 Predict Cancellation Risk", use_container_width=True)
        
        if submitted:
            # Create input data
            input_data = {
                'online_booking': online_booking,
                'mobile_site_booking': mobile_site_booking,
                'booking_hour': booking_hour,
                'is_weekend_booking': is_weekend_booking,
                'is_travel_type_1': 1 if travel_type == 1 else 0,
                'is_travel_type_2': 1 if travel_type == 2 else 0,
                'is_travel_type_3': 1 if travel_type == 3 else 0,
                'is_business_travel': is_business_travel,
                'is_leisure_travel': is_leisure_travel,
                'business_advance_booking': business_advance_booking,
                'leisure_last_minute': leisure_last_minute,
                'is_model_12': is_model_12,
                'trip_distance': trip_distance,
                'from_city_id_15.0': 1 if from_city_id == 15.0 else 0,
                'from_city_id_25.0': 1 if from_city_id == 25.0 else 0,
                'from_city_id_35.0': 1 if from_city_id == 35.0 else 0,
                'from_city_id_5.0': 1 if from_city_id == 5.0 else 0,
            }
            
            # Add missing features with default values
            for feature in metadata['feature_names']:
                if feature not in input_data:
                    input_data[feature] = 0
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])[metadata['feature_names']]
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 HIGH CANCELLATION RISK")
                    else:
                        st.success("✅ LOW CANCELLATION RISK")
                
                with col2:
                    st.metric("Cancellation Probability", f"{probability[1]:.1%}")
                
                with col3:
                    st.metric("No Cancellation Probability", f"{probability[0]:.1%}")
                
                # Risk interpretation
                if probability[1] > 0.7:
                    st.warning("⚠️ **High Risk**: Consider implementing retention strategies")
                elif probability[1] > 0.4:
                    st.info("⚡ **Medium Risk**: Monitor this booking closely")
                else:
                    st.success("✨ **Low Risk**: Booking likely to proceed smoothly")
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
