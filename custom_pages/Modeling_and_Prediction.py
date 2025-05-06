import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
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
        </style>
        """,
        unsafe_allow_html=True
    )

def train_model(model_name, X_train, y_train):
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "XGBoost Regressor": xgb.XGBRegressor(),  # Replace AdaBoost with XGBoost
        "Ridge Regression": Ridge()
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    return model

def run():
    add_custom_css()
    st.title("Model Training & Evaluation")
    st.markdown("### Select Regression Model")
    
    model_name = st.selectbox("Choose a model:", [
        "Gradient Boosting Regressor", 
        "Random Forest Regressor", 
        "Linear Regression", 
        "XGBoost Regressor",  # Include XGBoost in the model selection
        "Ridge Regression"
    ])

    df = pd.read_csv("merged_data.csv")
    st.markdown("#### Dataset Overview")
    st.write(df.head())
    
    # Exploratory Data Analysis (EDA)
    st.markdown("### Exploratory Data Analysis (EDA)")
    
    numeric_columns = ["year", "month", "day", "hour", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    categorical_columns = ["wd", "station", "Category"]
    
    st.markdown("#### Summary Statistics")
    st.write(df.describe())
    
    for col in numeric_columns:
        st.markdown(f"### Distribution of {col}")
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], bins=30, kde=True, color="blue")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        st.pyplot(plt)
    
    for col in categorical_columns:
        st.markdown(f"### Distribution of {col}")
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=col, palette="coolwarm")
        plt.xticks(rotation=45)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        st.pyplot(plt)
    
    # Correlation Matrix
    st.markdown("### Correlation Matrix")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    st.pyplot(plt)
    
    # Scatter Plots
    st.markdown("### Scatter Plots")
    for col in numeric_columns:
        if col != "PM2.5":
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=df[col], y=df["PM2.5"], alpha=0.5, color="red")
            plt.xlabel(col)
            plt.ylabel("PM2.5")
            plt.title(f"Scatter Plot: {col} vs PM2.5")
            st.pyplot(plt)
    
    # Box Plots
    st.markdown("### Box Plots")
    for col in numeric_columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(y=df[col], color="green")
        plt.ylabel(col)
        plt.title(f"Box Plot of {col}")
        st.pyplot(plt)
    
    # Data Preparation
    X = df[["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]]
    y = df["PM2.5"]
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = train_model(model_name, X_train, y_train)
    
    st.markdown(f"#### Training {model_name}")
    st.write(f"{model_name} trained successfully!")
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    st.markdown("### Model Performance")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    
    if hasattr(model, "feature_importances_"):
        st.markdown("### Feature Importance")
        feature_importance = model.feature_importances_
        feature_names = X.columns
        feature_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
        feature_df = feature_df.sort_values(by="Importance", ascending=False)
        st.dataframe(feature_df)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_df["Feature"], feature_df["Importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        st.pyplot(plt)
    
if __name__ == "__main__":
    run()
