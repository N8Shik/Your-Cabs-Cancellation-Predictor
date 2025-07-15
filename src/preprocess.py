import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Clean and prepare the YourCabs dataset for machine learning.
    
    This function:
    - Removes unnecessary columns
    - Handles missing values
    - Creates useful features from dates
    - Prepares data for modeling
    """
    # Make a copy so we don't change the original data
    df = df.copy()
    
    print(f"Starting with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Remove ID columns (they don't help predict cancellations)
    df = df.drop(columns=['id', 'user_id'], errors='ignore')
    
    # Fix 'NULL' strings - convert them to actual missing values
    if 'package_id' in df.columns:
        df['package_id'] = df['package_id'].replace('NULL', np.nan)
    
    # Convert date columns to proper datetime format
    df['from_date'] = pd.to_datetime(df['from_date'], errors='coerce')
    df['booking_created'] = pd.to_datetime(df['booking_created'], errors='coerce')
    
    # Create useful time-based features
    if 'from_date' in df.columns and 'booking_created' in df.columns:
        # How many hours between booking and travel?
        time_diff = df['from_date'] - df['booking_created']
        df['hours_advance'] = time_diff.dt.total_seconds() / 3600
        
        # Categorize booking types
        def get_booking_type(hours):
            if pd.isna(hours):
                return 'Unknown'
            elif hours <= 1:
                return 'LastMinute'  # Very urgent
            elif hours <= 6:
                return 'SameDay'     # Same day booking
            elif hours <= 24:
                return 'NextDay'     # Next day booking
            else:
                return 'Advanced'    # Planned ahead
        
        df['booking_type'] = df['hours_advance'].apply(get_booking_type)
        
        # What time of day was the booking made?
        df['booking_hour'] = df['booking_created'].dt.hour
        df['is_weekend_booking'] = (df['booking_created'].dt.dayofweek >= 5).astype(int)
    
    # Handle missing city IDs - fill with the most common value
    for col in ['from_city_id', 'to_city_id']:
        if col in df.columns and df[col].isna().any():
            most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else -1
            df[col] = df[col].fillna(most_common)
    
    # Handle travel type specific logic
    if 'travel_type_id' in df.columns:
        df = create_travel_type_features(df)
    
    # Vehicle model: create feature for the most popular model (usually model 12)
    if 'vehicle_model_id' in df.columns:
        # Fill missing values with the most common model
        if df['vehicle_model_id'].isna().any():
            most_common_model = df['vehicle_model_id'].mode()[0] if len(df['vehicle_model_id'].mode()) > 0 else 12
            df['vehicle_model_id'] = df['vehicle_model_id'].fillna(most_common_model)
        
        # Create binary feature for most popular model
        most_popular = df['vehicle_model_id'].mode()[0]
        df[f'is_model_{most_popular}'] = (df['vehicle_model_id'] == most_popular).astype(int)
    
    # Calculate trip distance if we have coordinates
    if all(col in df.columns for col in ['from_lat', 'from_long', 'to_lat', 'to_long']):
        df['trip_distance'] = calculate_distance(
            df['from_lat'], df['from_long'], 
            df['to_lat'], df['to_long']
        )
    
    # Remove columns we don't need anymore
    columns_to_drop = [
        'vehicle_model_id',  # We created a binary feature instead
        'from_date', 'booking_created',  # We extracted useful info from these
        'hours_advance',  # We have the categorical version
        'from_lat', 'from_long', 'to_lat', 'to_long',  # We calculated distance
        'package_id',  # Usually not very predictive
        'travel_type_id',  # We created specific travel type features
        'from_area_id', 'to_area_id'  # We have city IDs which are better
    ]
    
    # Only drop columns that actually exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    # Convert categorical columns to dummy variables
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or col in ['from_city_id', 'to_city_id']:
            categorical_cols.append(col)
    
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Handle any remaining missing values
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)  # For dummy variables
    
    print(f"Finished with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using the Haversine formula.
    Returns distance in kilometers.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    return 6371 * c


def create_travel_type_features(df):
    """
    Create travel type specific features based on travel_type_id.
    
    Different travel types have different cancellation patterns:
    - Type 1: Often airport/business trips (lower cancellation)
    - Type 2: Regular city trips (moderate cancellation) 
    - Type 3: Leisure/flexible trips (higher cancellation)
    """
    # Fill missing travel types with the most common type
    if df['travel_type_id'].isna().any():
        most_common_type = df['travel_type_id'].mode()[0] if len(df['travel_type_id'].mode()) > 0 else 2
        df['travel_type_id'] = df['travel_type_id'].fillna(most_common_type)
    
    # Create binary features for each travel type
    for travel_type in df['travel_type_id'].unique():
        if not pd.isna(travel_type):
            df[f'is_travel_type_{int(travel_type)}'] = (df['travel_type_id'] == travel_type).astype(int)
    
    # Create travel type specific features
    df['is_business_travel'] = (df['travel_type_id'] == 1).astype(int)  # Assuming type 1 is business
    df['is_leisure_travel'] = (df['travel_type_id'] == 3).astype(int)   # Assuming type 3 is leisure
    
    # Travel type specific booking patterns
    # Business travelers tend to book more in advance
    if 'hours_advance' in df.columns:
        df['business_advance_booking'] = df['is_business_travel'] * df['hours_advance']
        df['leisure_last_minute'] = df['is_leisure_travel'] * (df['hours_advance'] <= 2).astype(int)
    
    return df
