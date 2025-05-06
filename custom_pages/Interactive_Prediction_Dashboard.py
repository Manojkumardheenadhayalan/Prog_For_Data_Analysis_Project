import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb  # Import XGBoost

def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: black;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00FF00;
        }
        a {
            color: #1E90FF;
        }
        .st-info {
            background-color: #333333;
            color: white;
        }
        button {
            background-color: #444444;
            color: white;
        }
        .good { color: #00FF00; font-weight: bold; }
        .moderate { color: #FFFF00; font-weight: bold; }
        .unhealthy-sensitive { color: #FFA500; font-weight: bold; }
        .unhealthy { color: #FF0000; font-weight: bold; }
        .very-unhealthy { color: #8B008B; font-weight: bold; }
        .hazardous { color: #800000; font-weight: bold; }
        </style>
        """,
        unsafe_allow_html=True
    )

def get_model(model_name):
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBoost Regressor": xgb.XGBRegressor(),  # Replace AdaBoost with XGBoost
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge()
    }
    return models[model_name]

def run():
    add_custom_css()
    st.title("Interactive Prediction Tool")

    df = pd.read_csv("merged_data.csv")

    feature_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    target_col = "PM2.5"

    X = df[feature_cols]
    y = df[target_col]
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # User model choice
    model_name = st.selectbox("Choose a regression model", [
        "Gradient Boosting Regressor",
        "Random Forest Regressor",
        "XGBoost Regressor",  # Include XGBoost in the model selection
        "Linear Regression",
        "Ridge Regression"
    ])

    model = get_model(model_name)
    model.fit(X_train, y_train)

    # Input form
    st.markdown("### Enter Feature Values")
    user_input = {}
    for feature in feature_cols:
        user_input[feature] = st.number_input(f"{feature}:", value=float(X[feature].mean()))

    if st.button("Predict PM2.5 Concentration"):
        user_data = np.array([list(user_input.values())])
        user_prediction = model.predict(user_data)
        pm25 = user_prediction[0]

        # Categorize air quality
        if pm25 <= 50:
            quality, quality_class = "Good", "good"
        elif pm25 <= 100:
            quality, quality_class = "Moderate", "moderate"
        elif pm25 <= 150:
            quality, quality_class = "Unhealthy for sensitive groups", "unhealthy-sensitive"
        elif pm25 <= 200:
            quality, quality_class = "Unhealthy", "unhealthy"
        elif pm25 <= 300:
            quality, quality_class = "Very Unhealthy", "very-unhealthy"
        else:
            quality, quality_class = "Hazardous", "hazardous"

        st.write(f"**Model Used:** {model_name}")
        st.write(f"**Predicted PM2.5 Concentration:** {pm25:.2f}")
        st.markdown(f"<div class='{quality_class}'>Air Quality: {quality}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()
