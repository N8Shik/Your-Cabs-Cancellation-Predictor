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
        "High Risk Business Trip": {
            'online_booking': 1,
            'mobile_site_booking': 0,
            'booking_hour': 22,
            'is_weekend_booking': 0,
            'travel_type': 1,
            'is_business_travel': 1,
            'is_leisure_travel': 0,
            'trip_distance': 25.5,
            'business_advance_booking': 0,
            'leisure_last_minute': 0,
            'is_model_12': 0,
            'from_city_id': 35.0
        },
        "Low Risk Leisure Trip": {
            'online_booking': 0,
            'mobile_site_booking': 1,
            'booking_hour': 14,
            'is_weekend_booking': 1,
            'travel_type': 2,
            'is_business_travel': 0,
            'is_leisure_travel': 1,
            'trip_distance': 8.2,
            'business_advance_booking': 0,
            'leisure_last_minute': 0,
            'is_model_12': 1,
            'from_city_id': 15.0
        },
        "Medium Risk Last-Minute": {
            'online_booking': 1,
            'mobile_site_booking': 1,
            'booking_hour': 18,
            'is_weekend_booking': 0,
            'travel_type': 3,
            'is_business_travel': 0,
            'is_leisure_travel': 1,
            'trip_distance': 12.0,
            'business_advance_booking': 0,
            'leisure_last_minute': 1,
            'is_model_12': 0,
            'from_city_id': 25.0
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
        - 🟢 **Low Risk** (0-40%): Likely to proceed
        - 🟡 **Medium Risk** (40-70%): Monitor closely  
        - 🔴 **High Risk** (70%+): Take action
        """)
        
        st.header("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC Score", f"{metadata['auc_score']:.1%}")
        with col2:
            st.metric("Accuracy", f"{metadata['accuracy']:.1%}")
        
        st.markdown("*Model trained on historical booking data*")
    
    # Example scenarios
    with st.expander("� Try Example Scenarios", expanded=False):
        st.markdown("**Click any example below to auto-fill the form:**")
        examples = get_example_scenarios()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔴 High Risk Example", use_container_width=True):
                st.session_state.update(examples["High Risk Business Trip"])
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
        with col2:
            if st.button("🟢 Low Risk Example", use_container_width=True):
                st.session_state.update(examples["Low Risk Leisure Trip"])
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
        with col3:
            if st.button("🟡 Medium Risk Example", use_container_width=True):
                st.session_state.update(examples["Medium Risk Last-Minute"])
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
    
    st.markdown("## 📊 Enter Booking Details")
    
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
                [5.0, 15.0, 25.0, 35.0], 
                format_func=lambda x: f"🏙️ City {int(x)} {'(Downtown)' if x == 15.0 else '(Suburb)' if x == 25.0 else '(Airport)' if x == 35.0 else '(Industrial)'}",
                index=[5.0, 15.0, 25.0, 35.0].index(st.session_state.get('from_city_id', 15.0)),
                help="Which city/area is the pickup location?"
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
                
                # Display results with enhanced UI
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                # Main prediction display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    prob_percent = probability[1] * 100
                    if prediction == 1:
                        st.error(f"🚨 **HIGH CANCELLATION RISK** ({prob_percent:.1f}%)")
                    else:
                        st.success(f"✅ **LOW CANCELLATION RISK** ({prob_percent:.1f}%)")
                
                # Detailed metrics
                st.markdown("### 📊 Detailed Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Cancellation Risk", 
                        f"{probability[1]:.1%}",
                        delta=f"{probability[1] - 0.5:.1%}" if probability[1] > 0.5 else f"{probability[1] - 0.5:.1%}"
                    )
                
                with col2:
                    st.metric(
                        "Success Probability", 
                        f"{probability[0]:.1%}",
                        delta=f"{probability[0] - 0.5:.1%}" if probability[0] > 0.5 else f"{probability[0] - 0.5:.1%}"
                    )
                
                with col3:
                    confidence = max(probability[0], probability[1])
                    st.metric("Model Confidence", f"{confidence:.1%}")
                
                with col4:
                    risk_level = "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.4 else "Low"
                    st.metric("Risk Level", risk_level)
                
                # Risk gauge visualization
                st.markdown("### 🎚️ Risk Gauge")
                gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 3, 1])
                with gauge_col2:
                    # Create a simple progress bar as risk gauge
                    risk_value = probability[1]
                    if risk_value < 0.4:
                        st.success(f"Risk Level: {risk_value:.1%}")
                        st.progress(risk_value, text="Low Risk Zone")
                    elif risk_value < 0.7:
                        st.warning(f"Risk Level: {risk_value:.1%}")
                        st.progress(risk_value, text="Medium Risk Zone")
                    else:
                        st.error(f"Risk Level: {risk_value:.1%}")
                        st.progress(risk_value, text="High Risk Zone")
                
                # Recommendations based on risk level
                st.markdown("### 💡 Recommendations")
                if probability[1] > 0.7:
                    st.error("""
                    **🚨 HIGH RISK - Immediate Action Required:**
                    - Contact customer to confirm booking
                    - Offer incentives or discounts
                    - Send confirmation reminders
                    - Consider alternative options
                    - Flag for priority customer service
                    """)
                elif probability[1] > 0.4:
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
    if st.button("🔄 Clear Form", help="Reset all inputs to default values"):
        keys_to_clear = ['online_booking', 'mobile_site_booking', 'booking_hour', 'is_weekend_booking', 
                        'travel_type', 'is_business_travel', 'is_leisure_travel', 'trip_distance',
                        'business_advance_booking', 'leisure_last_minute', 'is_model_12', 'from_city_id']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🚖 YourCabs Cancellation Predictor | Built with Streamlit & XGBoost | 
        <a href='https://github.com/N8Shik/Your-Cabs-Cancellation-Predictor' target='_blank'>View Source Code</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
