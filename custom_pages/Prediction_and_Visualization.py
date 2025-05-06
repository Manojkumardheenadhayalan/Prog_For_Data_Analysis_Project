import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Core Machine Learning Libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Advanced Regression Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

# Deep Learning Models (Optional)
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

# Custom CSS and Styling
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stDataFrame {
        background-color: #262730;
        color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #1DB954;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Advanced Model Training Function
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'Accuracy (%)': round(r2 * 100, 2)
    }

# Model Zoo
def get_regression_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100),
        "LightGBM Regressor": lgb.LGBMRegressor(n_estimators=100),
        "CatBoost Regressor": cb.CatBoostRegressor(verbose=0, n_estimators=100)
    }
    
    # Conditionally add TabNet if available
    if TABNET_AVAILABLE:
        models["TabNet Regressor"] = TabNetRegressor()
    
    return models

# Main Streamlit Application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Advanced Regression Model Playground",
        page_icon="📊",
        layout="wide"
    )
    
    # Apply custom CSS
    add_custom_css()
    
    # Title and Description
    st.title("🚀 Advanced Regression Model Playground")
    st.markdown("Comparative Analysis of Multiple Regression Techniques")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display Dataset Information
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Dataset Shape: {df.shape}")
            st.dataframe(df.head())
        
        with col2:
            st.write("Column Statistics")
            st.dataframe(df.describe())
        
        # Feature Selection
        st.subheader("Feature Selection")
        all_columns = df.columns.tolist()
        feature_columns = st.multiselect("Select Features", all_columns)
        target_column = st.selectbox("Select Target Variable", all_columns)
        
        if feature_columns and target_column:
            # Prepare Data
            X = df[feature_columns]
            y = df[target_column]
            
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Model Comparison
            st.subheader("Model Performance Comparison")
            models = get_regression_models()
            
            # Comparison DataFrame
            comparison_results = []
            
            for name, model in models.items():
                try:
                    result = train_and_evaluate_model(
                        model, 
                        X_train_scaled, 
                        X_test_scaled, 
                        y_train, 
                        y_test
                    )
                    result['Model'] = name
                    comparison_results.append(result)
                except Exception as e:
                    st.error(f"Error with {name}: {str(e)}")
            
            # Display Comparison Table
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df = comparison_df.sort_values('Accuracy (%)', ascending=False)
            st.dataframe(comparison_df)
            
            # Model Selection for Deep Dive
            st.subheader("Model Deep Dive")
            selected_model_name = st.selectbox(
                "Select Model for Detailed Analysis", 
                comparison_df['Model'].tolist()
            )
            
            # Visualization Options
            if st.checkbox("Show Model Insights"):
                selected_model = models[selected_model_name]
                selected_model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = selected_model.predict(X_test_scaled)
                
                # Scatter Plot
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title(f"{selected_model_name}: Actual vs Predicted")
                st.pyplot(plt)

# Run the application
if __name__ == "__main__":
    main()