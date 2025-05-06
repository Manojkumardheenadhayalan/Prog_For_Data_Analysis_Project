import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb  # Import XGBoost

# Add custom CSS for styling
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

# Function to train the model
def train_model(model_name, X_train, y_train):
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "XGBoost Regressor": xgb.XGBRegressor(),
        "Ridge Regression": Ridge()
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    return model

# Main function to run the app
def run():
    # Apply custom CSS
    add_custom_css()
    st.title("Air Quality Prediction & Visualization")

    # Model comparison section
    df = pd.read_csv("merged_data.csv")
    st.markdown("#### Dataset Preview")
    st.dataframe(df.head())

    feature_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    target_col = "PM2.5"

    # Prepare the data
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train all models and evaluate them
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "XGBoost Regressor": xgb.XGBRegressor(),
        "Ridge Regression": Ridge()
    }

    comparison_data = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        comparison_data.append({
            "Model": name,
            "MSE": round(mse, 2),
            "R²": round(r2, 2),
            "Accuracy (%)": round(r2 * 100, 2)
        })

    # Display model comparison table
    st.markdown("### 🔍 Model Comparison Overview")
    st.dataframe(pd.DataFrame(comparison_data).sort_values(by="Accuracy (%)", ascending=False), use_container_width=True)

    # Select model for detailed visualization
    model_name = st.selectbox("### 📊 Choose a model for detailed visualization", list(models.keys()), label_visibility="hidden")
    model = models[model_name]
    predictions = model.predict(X_test)

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    corr = df[feature_cols + [target_col]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(plt)

    # Model Performance Metrics
    st.markdown("### Model Performance")
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    accuracy = r2 * 100

    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    st.write(f"**Model Accuracy:** {accuracy:.2f}%")

    # Sample Predictions (first 10)
    st.markdown("### Sample Predictions (First 10)")
    results_df = pd.DataFrame({
        "Actual PM2.5": y_test[:10].values,
        "Predicted PM2.5": predictions[:10]
    }).reset_index(drop=True)
    st.dataframe(results_df)

    # Actual vs Predicted scatter plot
    st.markdown("### Visualization: Actual vs Predicted Scatter")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color="blue", label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect Prediction")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.legend()
    st.pyplot(plt)

    # Actual vs Predicted Line plot
    st.markdown("### Visualization: Actual vs Predicted Line Plot")
    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:50].values, label="Actual", marker='o')
    plt.plot(predictions[:50], label="Predicted", marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("PM2.5 Value")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Additional Visualizations

    # Model Comparison Bar Chart
    st.markdown("### Model Comparison: Accuracy Bar Chart")
    model_names = [entry["Model"] for entry in comparison_data]
    accuracies = [entry["Accuracy (%)"] for entry in comparison_data]
    plt.figure(figsize=(10, 6))
    plt.barh(model_names, accuracies, color='green')
    plt.xlabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    st.pyplot(plt)

    # Feature Distribution (for all features)
    st.markdown("### Feature Distribution")
    for feature in feature_cols:
        st.markdown(f"#### Distribution of {feature}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30, color='purple')
        plt.title(f"Distribution of {feature}")
        st.pyplot(plt)

    # Predicted vs Actual PM2.5 Distribution
    st.markdown("### Predicted vs Actual PM2.5 Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test, kde=True, color='blue', label='Actual PM2.5')
    sns.histplot(predictions, kde=True, color='red', label='Predicted PM2.5')
    plt.legend()
    plt.title("Distribution of Actual vs Predicted PM2.5")
    st.pyplot(plt)

    # Learning Curve (for selected model)
    from sklearn.model_selection import learning_curve
    st.markdown("### Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score", color="blue")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation Score", color="red")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Learning Curve")
    st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    run()
