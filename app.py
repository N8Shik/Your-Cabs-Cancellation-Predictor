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

def get_example_scenarios():
    """Return example booking scenarios for users to try"""
    return {
        "Highest Risk Pattern": {
            'online_booking': 1,
            'mobile_site_booking': 0,
            'booking_hour': 1,  # Very early morning (1 AM)
            'is_weekend_booking': 1,  # Weekend
            'travel_type': 1,  # Business type but no specific flags
            'is_business_travel': 0,
            'is_leisure_travel': 0,
            'trip_distance': 10.0,  # Normal distance
            'business_advance_booking': 0,
            'leisure_last_minute': 0,
            'is_model_12': 0,  # Standard vehicle
            'from_city_id': 15.0  # Downtown pickup
        },
        "Medium Risk Pattern": {
            'online_booking': 1,
            'mobile_site_booking': 0,
            'booking_hour': 14,  # Afternoon
            'is_weekend_booking': 0,  # Weekday
            'travel_type': 1,  # Business
            'is_business_travel': 1,
            'is_leisure_travel': 0,
            'trip_distance': 10.0,  # Normal distance
            'business_advance_booking': 1,  # Advance booking
            'leisure_last_minute': 0,
            'is_model_12': 1,  # Premium vehicle
            'from_city_id': 15.0  # Downtown pickup
        },
        "Lowest Risk Pattern": {
            'online_booking': 1,
            'mobile_site_booking': 0,
            'booking_hour': 14,  # Afternoon
            'is_weekend_booking': 0,  # Weekday
            'travel_type': 2,  # Leisure
            'is_business_travel': 0,
            'is_leisure_travel': 1,
            'trip_distance': 50.0,  # Long leisure trip
            'business_advance_booking': 0,
            'leisure_last_minute': 1,  # Last minute leisure
            'is_model_12': 0,  # Standard vehicle
            'from_city_id': 15.0  # Downtown pickup
        }
    }

def main():
    st.title("🚖 YourCabs Cancellation Predictor")
    st.markdown("""
    ### 🎯 Predict booking cancellation risk with AI
    This tool uses machine learning to analyze booking patterns and predict the likelihood of cab cancellations.
    Perfect for cab companies to optimize their operations and improve customer retention.
    """)
    st.markdown("---")
    
    # Load model
    model, metadata = load_model()
    
    if model is None:
        st.error("Could not load the model. Please check if model files exist.")
        return
    
    # Sidebar with helpful information
    with st.sidebar:
        st.header("📋 Quick Guide")
        st.markdown("""
        **How to use:**
        1. Fill in the booking details
        2. Click "Predict" to get risk assessment
        3. Review the results and recommendations
        
        **Risk Levels:**
        - 🟢 **Low Risk** (0-3%): Very likely to proceed
        - 🟡 **Medium Risk** (3-5%): Monitor closely  
        - 🔴 **High Risk** (5%+): Requires attention
        
        ⚠️ **Important Note:** This model was trained using SMOTE (Synthetic Minority Oversampling) to handle class imbalance. The original data had only 7.2% cancellations, but SMOTE balanced this to 50% for better risk detection. **The model predicts realistic ranges (2-6%) - focus on relative differences between predictions.**
        """)
        
        st.header("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC Score", f"{metadata['auc_score']:.1%}")
        with col2:
            st.metric("CV AUC", f"{metadata.get('cv_auc_score', metadata['auc_score']):.1%}")
        
        # Show training method
        training_method = metadata.get('training_method', 'Original')
        if training_method == 'SMOTE':
            st.success("🎯 **SMOTE Balanced Model** - Enhanced for better risk detection")
        else:
            st.info("📊 **Standard Model**")
        
        st.markdown("*Model trained on historical booking data*")
    
    # Example scenarios
    with st.expander("🔍 Try Example Scenarios", expanded=False):
        st.markdown("**Click any example below to auto-fill the form:**")
        examples = get_example_scenarios()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔴 Highest Risk Example", use_container_width=True, key="high_risk_btn"):
                for key, value in examples["Highest Risk Pattern"].items():
                    st.session_state[key] = value
                st.success("✅ Highest risk pattern loaded!")
        with col2:
            if st.button("🟢 Lowest Risk Example", use_container_width=True, key="low_risk_btn"):
                for key, value in examples["Lowest Risk Pattern"].items():
                    st.session_state[key] = value
                st.success("✅ Lowest risk pattern loaded!")
        with col3:
            if st.button("🟡 Medium Risk Example", use_container_width=True, key="medium_risk_btn"):
                for key, value in examples["Medium Risk Pattern"].items():
                    st.session_state[key] = value
                st.success("✅ Medium risk pattern loaded!")
        
        # Show current example if any is loaded
        if any(key in st.session_state for key in ['online_booking', 'mobile_site_booking', 'booking_hour']):
            st.info("💡 **Tip:** Example values are now loaded in the form below. You can modify them or click 'Predict' to see results.")
    
    st.markdown("## 📊 Enter Booking Details")
    
    # Show which example is currently loaded
    if 'online_booking' in st.session_state and 'trip_distance' in st.session_state:
        # Detect which example is loaded based on distinctive values
        current_distance = st.session_state.get('trip_distance', 0)
        current_hour = st.session_state.get('booking_hour', 12)
        current_city = st.session_state.get('from_city_id', 15.0)
        
        if current_distance == 55.0 and current_hour == 2 and current_city == 31.0:
            st.info("🔴 **Highest Risk Pattern** example is currently loaded")
        elif current_distance == 8.5 and current_hour == 14 and current_city == 15.0:
            st.success("🟢 **Lowest Risk Pattern** example is currently loaded")
        elif current_distance == 25.0 and current_hour == 23 and current_city == 15.0:
            st.warning("🟡 **Medium Risk Pattern** example is currently loaded")
        else:
            st.info("📝 **Custom values** are loaded in the form")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📱 Booking Method")
            st.markdown("*How was the booking made?*")
            online_booking = st.selectbox(
                "Online Booking", 
                [0, 1], 
                format_func=lambda x: "✅ Yes" if x else "❌ No",
                index=st.session_state.get('online_booking', 0),
                help="Was this booking made through an online platform?"
            )
            mobile_site_booking = st.selectbox(
                "Mobile Site Booking", 
                [0, 1], 
                format_func=lambda x: "📱 Yes" if x else "💻 No",
                index=st.session_state.get('mobile_site_booking', 0),
                help="Was this booking made through a mobile website/app?"
            )
            
            st.subheader("⏰ Timing Details")
            st.markdown("*When was the booking made?*")
            booking_hour = st.slider(
                "Booking Hour (24h format)", 
                0, 23, 
                st.session_state.get('booking_hour', 12),
                help="Hour when the booking was made (0 = midnight, 23 = 11 PM)"
            )
            st.caption(f"Selected time: {booking_hour:02d}:00 ({'Morning' if 6 <= booking_hour < 12 else 'Afternoon' if 12 <= booking_hour < 18 else 'Evening' if 18 <= booking_hour < 22 else 'Night'})")
            
            is_weekend_booking = st.selectbox(
                "Weekend Booking", 
                [0, 1], 
                format_func=lambda x: "🎉 Weekend" if x else "📅 Weekday",
                index=st.session_state.get('is_weekend_booking', 0),
                help="Was this booking made on a weekend?"
            )
            
            st.subheader("🚗 Trip Details")
            st.markdown("*What type of trip is this?*")
            travel_type = st.selectbox(
                "Travel Type", 
                [1, 2, 3], 
                format_func=lambda x: f"Type {x} {'(Business)' if x == 1 else '(Leisure)' if x == 2 else '(Other)'}",
                index=st.session_state.get('travel_type', 1) - 1,
                help="Type 1: Business, Type 2: Leisure, Type 3: Other"
            )
            
            trip_distance = st.number_input(
                "Trip Distance (km)", 
                min_value=0.0, 
                max_value=100.0, 
                value=float(st.session_state.get('trip_distance', 5.0)), 
                step=0.1,
                help="Distance of the planned trip in kilometers"
            )
            distance_category = "Short" if trip_distance < 5 else "Medium" if trip_distance < 15 else "Long"
            st.caption(f"Distance category: {distance_category} trip")
        
        with col2:
            st.subheader("💼 Travel Purpose")
            st.markdown("*Business or leisure travel?*")
            is_business_travel = st.selectbox(
                "Business Travel", 
                [0, 1], 
                format_func=lambda x: "💼 Yes" if x else "❌ No",
                index=st.session_state.get('is_business_travel', 0),
                help="Is this for business purposes?"
            )
            is_leisure_travel = st.selectbox(
                "Leisure Travel", 
                [0, 1], 
                format_func=lambda x: "🏖️ Yes" if x else "❌ No",
                index=st.session_state.get('is_leisure_travel', 0),
                help="Is this for leisure/vacation purposes?"
            )
            
            st.subheader("📅 Booking Pattern")
            st.markdown("*How far in advance was this booked?*")
            business_advance_booking = st.selectbox(
                "Business Advance Booking", 
                [0, 1], 
                format_func=lambda x: "📋 Yes" if x else "❌ No",
                index=st.session_state.get('business_advance_booking', 0),
                help="Business trip booked well in advance?"
            )
            leisure_last_minute = st.selectbox(
                "Leisure Last Minute", 
                [0, 1], 
                format_func=lambda x: "⚡ Yes" if x else "❌ No",
                index=st.session_state.get('leisure_last_minute', 0),
                help="Leisure trip booked at the last minute?"
            )
            
            st.subheader("🚙 Vehicle & Location")
            st.markdown("*Vehicle and pickup details*")
            is_model_12 = st.selectbox(
                "Premium Vehicle (Model 12)", 
                [0, 1], 
                format_func=lambda x: "⭐ Premium" if x else "🚗 Standard",
                index=st.session_state.get('is_model_12', 0),
                help="Is this a premium vehicle model?"
            )
            from_city_id = st.selectbox(
                "Pickup City", 
                [15.0, 31.0], 
                format_func=lambda x: f"🏙️ City {int(x)} {'(Downtown)' if x == 15.0 else '(Suburb)' if x == 31.0 else '(Other)'}",
                index=[15.0, 31.0].index(st.session_state.get('from_city_id', 15.0)),
                help="Which city/area is the pickup location? (Limited to cities with sufficient training data)"
            )
        
        # Form validation and submission
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "🔮 Predict Cancellation Risk", 
                use_container_width=True,
                type="primary"
            )
        
        # Input validation
        if submitted:
            validation_errors = []
            if is_business_travel and is_leisure_travel:
                validation_errors.append("⚠️ Trip cannot be both business AND leisure")
            if business_advance_booking and not is_business_travel:
                validation_errors.append("⚠️ Business advance booking requires business travel to be selected")
            if leisure_last_minute and not is_leisure_travel:
                validation_errors.append("⚠️ Leisure last minute requires leisure travel to be selected")
            
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                st.stop()
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
                'from_city_id_31.0': 1 if from_city_id == 31.0 else 0,
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
                
                # Convert numpy types to Python types to avoid Streamlit compatibility issues
                prediction = int(prediction)
                probability = [float(p) for p in probability]
                
                # Display results with enhanced UI
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                # Main prediction display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    prob_percent = probability[1] * 100
                    if prob_percent >= 5.0:
                        st.error(f"🚨 **HIGH CANCELLATION RISK** ({prob_percent:.1f}%)")
                    elif prob_percent >= 3.0:
                        st.warning(f"⚡ **MEDIUM CANCELLATION RISK** ({prob_percent:.1f}%)")
                    else:
                        st.success(f"✅ **LOW CANCELLATION RISK** ({prob_percent:.1f}%)")
                
                # Important model limitation notice
                st.info("ℹ️ **SMOTE Model:** This model was trained with balanced data (50% cancellations) using SMOTE oversampling. It predicts realistic cancellation rates (2-6%) with excellent risk differentiation - a 6% prediction indicates 3x higher risk than 2%.")
                
                # Detailed metrics
                st.markdown("### 📊 Detailed Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    delta_value = probability[1] - 0.5
                    st.metric(
                        "Cancellation Risk", 
                        f"{probability[1]:.1%}",
                        delta=f"{delta_value:.1%}"
                    )
                
                with col2:
                    delta_value = probability[0] - 0.5
                    st.metric(
                        "Success Probability", 
                        f"{probability[0]:.1%}",
                        delta=f"{delta_value:.1%}"
                    )
                
                with col3:
                    confidence = max(probability[0], probability[1])
                    st.metric("Model Confidence", f"{confidence:.1%}")
                
                with col4:
                    if probability[1] > 0.05:
                        risk_level = "High"
                    elif probability[1] > 0.03:
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"
                    st.metric("Risk Level", risk_level)
                
                # Risk gauge visualization
                st.markdown("### 🎚️ Risk Gauge")
                gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 3, 1])
                with gauge_col2:
                    # Adjusted risk gauge for SMOTE model actual ranges (0-6%)
                    risk_value = float(probability[1])
                    if risk_value < 0.03:
                        st.success(f"Risk Level: {risk_value:.1%}")
                        st.progress(min(risk_value * 16.67, 1.0), text="Low Risk Zone")  # Scale 0-3% to 0-50%
                    elif risk_value < 0.05:
                        st.warning(f"Risk Level: {risk_value:.1%}")
                        st.progress(min(risk_value * 16.67, 1.0), text="Medium Risk Zone")  # Scale 3-5% to 50-83%
                    else:
                        st.error(f"Risk Level: {risk_value:.1%}")
                        st.progress(min(risk_value * 16.67, 1.0), text="High Risk Zone")  # Scale 5%+ to 83%+
                
                # Recommendations based on risk level
                st.markdown("### 💡 Recommendations")
                if probability[1] > 0.05:
                    st.error("""
                    **🚨 HIGH RISK - Take Action:**
                    - Contact customer to confirm booking
                    - Offer incentives or discounts
                    - Send confirmation reminders
                    - Consider alternative options
                    - Flag for priority customer service
                    """)
                elif probability[1] > 0.03:
                    st.warning("""
                    **⚡ MEDIUM RISK - Monitor Closely:**
                    - Send booking confirmation
                    - Monitor for any changes
                    - Prepare backup options
                    - Send reminder 24h before trip
                    """)
                else:
                    st.success("""
                    **✨ LOW RISK - Proceed Normally:**
                    - Standard confirmation process
                    - Regular service delivery
                    - Customer likely to show up
                    - Low priority for intervention
                    """)
                
                # Additional insights
                st.markdown("### 🔍 Key Factors Analysis")
                factors_col1, factors_col2 = st.columns(2)
                
                with factors_col1:
                    st.markdown("**Risk Factors:**")
                    risk_factors = []
                    if booking_hour >= 22 or booking_hour <= 5:
                        risk_factors.append("🌙 Late night/early morning booking")
                    if trip_distance > 20:
                        risk_factors.append("🛣️ Long distance trip")
                    if leisure_last_minute:
                        risk_factors.append("⚡ Last-minute leisure booking")
                    if from_city_id == 35.0:
                        risk_factors.append("✈️ Airport pickup location")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("• No major risk factors identified")
                
                with factors_col2:
                    st.markdown("**Positive Factors:**")
                    positive_factors = []
                    if is_model_12:
                        positive_factors.append("⭐ Premium vehicle selected")
                    if business_advance_booking:
                        positive_factors.append("📋 Advanced business booking")
                    if 9 <= booking_hour <= 17:
                        positive_factors.append("🕘 Business hours booking")
                    if is_weekend_booking and is_leisure_travel:
                        positive_factors.append("🎉 Weekend leisure trip")
                    
                    if positive_factors:
                        for factor in positive_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("• Standard booking profile")
                        
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
                st.markdown("**Please check your inputs and try again.**")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("### 📚 About This Tool")
    
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        This AI model analyzes historical booking patterns to predict cancellation risk. It considers:
        
        **Key Factors:**
        - **Booking Method**: Online vs mobile preferences
        - **Timing Patterns**: When bookings are made vs when trips occur
        - **Travel Purpose**: Business vs leisure travel behaviors  
        - **Distance**: Short, medium, and long-distance trip patterns
        - **Location**: Different pickup areas have different cancellation rates
        - **Vehicle Type**: Premium vs standard vehicle preferences
        
        **Model Performance:**
        - Trained on thousands of historical bookings
        - {accuracy:.1%} accuracy rate
        - {auc:.1%} AUC score (industry standard metric)
        - Regularly updated with new data
        
        **⚠️ Important Limitations:**
        - The model was trained on data where only 7.2% of bookings were cancelled
        - **SMOTE oversampling** was used to create balanced training data (50% cancellations)
        - This improves risk detection and provides realistic prediction ranges (1-3%)
        - **Focus on relative differences** between predictions rather than absolute percentages
        - A 3% prediction vs 1% prediction indicates 3x higher risk - this is significant!
        - The model excels at ranking risk levels rather than predicting exact cancellation rates
        """.format(accuracy=metadata['accuracy'], auc=metadata['auc_score']))
    
    with st.expander("🎯 Use Cases", expanded=False):
        st.markdown("""
        **For Cab Companies:**
        - Optimize driver allocation
        - Reduce no-show incidents  
        - Improve customer retention
        - Plan backup vehicles
        
        **For Operations Teams:**
        - Prioritize customer outreach
        - Implement dynamic pricing
        - Monitor booking quality
        - Reduce operational costs
        """)
    
    # Clear form button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Clear Form", help="Reset all inputs to default values", use_container_width=True):
            keys_to_clear = ['online_booking', 'mobile_site_booking', 'booking_hour', 'is_weekend_booking', 
                            'travel_type', 'is_business_travel', 'is_leisure_travel', 'trip_distance',
                            'business_advance_booking', 'leisure_last_minute', 'is_model_12', 'from_city_id']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Form cleared! All inputs reset to default values.")
            st.info("💡 Refresh the page or scroll up to see the reset form.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🚖 YourCabs Cancellation Predictor | Built with Streamlit & XGBoost | 
        <a href='https://github.com/N8Shik/Your-Cabs-Cancellation-Predictor' target='_blank'>View Source Code</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
