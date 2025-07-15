# 🚖 YourCabs Cancellation Predictor

A machine learning project that predicts cab cancellation probability using various booking and travel features.

## 🎯 Project Overview

This project uses XGBoost to predict whether a cab booking will be cancelled based on:
- Booking details (online/mobile, timing, weekend)
- Travel type and patterns
- Distance and location
- Historical booking behavior

**Model Performance:**
- **AUC Score:** 0.793
- **Accuracy:** 92.9%
- **Best Model:** XGBoost

## 🚀 Live Demo

**[Try the Streamlit App](https://your-app-url.streamlit.app)** *(will be available after deployment)*

## 📁 Project Structure

```
Your-Cabs-Cancellation-Predictor/
├── app.py                 # Streamlit web application
├── data/
│   └── YourCabs.csv      # Raw dataset
├── models/
│   ├── best_model_xgboost.joblib     # Trained XGBoost model
│   └── model_metadata_xgboost.json  # Model metadata
├── notebooks/
│   └── model_training.ipynb         # Complete ML pipeline
├── src/
│   └── preprocess.py               # Data preprocessing functions
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/N8Shik/Your-Cabs-Cancellation-Predictor.git
   cd Your-Cabs-Cancellation-Predictor
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. **Fork this repository**
2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Set main file path: `app.py`
   - Deploy!

## 📊 Model Training

The complete model training process is in `notebooks/model_training.ipynb` which includes:

- **Data Loading & Preprocessing**
- **Model Comparison** (Linear Regression, Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **Performance Evaluation**
- **Feature Importance Analysis**
- **Model Saving**

### Models Tested:
| Model | AUC Score | Accuracy |
|-------|-----------|----------|
| XGBoost | 0.793 | 92.9% |
| Random Forest | 0.785 | 71.5% |
| Logistic Regression | 0.771 | 69.0% |
| Decision Tree | 0.769 | 75.9% |

## 🔧 Usage

### Web App
1. Open the Streamlit app
2. Fill in booking details
3. Get instant cancellation risk prediction

### Programmatic Usage
```python
import joblib
import json
import pandas as pd

# Load model and metadata
model = joblib.load('./models/best_model_xgboost.joblib')
with open('./models/model_metadata_xgboost.json', 'r') as f:
    metadata = json.load(f)

# Make prediction
prediction = model.predict(your_data)
probability = model.predict_proba(your_data)
```

## 📈 Key Features

- **Real-time Predictions:** Instant cancellation risk assessment
- **High Accuracy:** 92.9% accuracy with AUC of 0.793
- **User-Friendly Interface:** Clean Streamlit web app
- **Comprehensive Analysis:** Feature importance and model comparison
- **Production Ready:** Proper model versioning and metadata

## 🧪 Model Features

The model uses these key features:
1. **Booking Method:** Online vs mobile booking
2. **Timing:** Booking hour and weekend patterns
3. **Travel Type:** Business vs leisure travel
4. **Distance:** Trip distance in kilometers
5. **Location:** From city ID
6. **Patterns:** Advance booking vs last-minute

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Shikhar Naithani** - [@N8Shik](https://github.com/N8Shik)

Project Link: [https://github.com/N8Shik/Your-Cabs-Cancellation-Predictor](https://github.com/N8Shik/Your-Cabs-Cancellation-Predictor)

---

**Made with ❤️ and ☕ by Shikhar**
