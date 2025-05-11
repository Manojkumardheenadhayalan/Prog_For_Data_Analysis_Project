import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# ====================== CUSTOM CSS ======================
def add_custom_css():
    st.markdown("""
        <style>
        body { background-color: black; color: white; }
        .stApp { background-color: black !important; color: white !important; }
        [data-testid="stSidebarContent"], [data-testid="stHeader"] { background-color: black; color: white; }
        .stButton > button { background-color: #333333; color: white; }
        h1, h2, h3, h4, h5, h6, p, div, span, label { color: white !important; }
        .stDataFrame, .stTable { background-color: #222222; color: white; }
        [role="radiogroup"] > div { background-color: #222222; margin-bottom: 5px; border-radius: 10px; padding: 5px; }
        [role="radiogroup"] label { color: white !important; }
        .good { color: #00FF00; font-weight: bold; }
        .moderate { color: #FFFF00; font-weight: bold; }
        .unhealthy-sensitive { color: #FFA500; font-weight: bold; }
        .unhealthy { color: #FF0000; font-weight: bold; }
        .very-unhealthy { color: #8B008B; font-weight: bold; }
        .hazardous { color: #800000; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

# ====================== PAGE 1: INTRODUCTION ======================
def page_introduction():
    add_custom_css()
    st.title("Introduction")
    st.markdown("""
        ##  Project Overview
        This project aims to predict the **PM2.5 concentration** in Beijing from 2013 to 2017 using advanced machine learning techniques. 
        Understanding PM2.5 levels is crucial for environmental planning and improving air quality.
        """)
    st.markdown("""
        ## Dataset Information
        - **Source**: Beijing Environmental Monitoring Center
        - **Time Period**: March 1, 2013, to February 28, 2017
        - **Features**: 
            - Air pollutant levels (PM2.5, PM10, SO2, NO2, CO, O3)
            - Meteorological data (Temperature, Pressure, Wind Speed, Rainfall)
        - **Sites**: Data from **12 monitoring stations** across Beijing.
        Missing values in the dataset are denoted as `NA` and have been handled appropriately during preprocessing.
        """)
    st.markdown("""
        ##  Key Objectives
        - Perform **exploratory data analysis (EDA)** to gain insights into PM2.5 trends and their correlation with other factors.
        - Build and evaluate **regression models** to predict PM2.5 concentrations.
        - Visualize results for better understanding and actionable insights.
        """)
    st.markdown("###  Why is PM2.5 Important?")
    if st.button("Learn More"):
        st.write("""
            PM2.5 particles are fine particulate matter with a diameter of less than 2.5 micrometers.
            - **Health Impact**: Can penetrate deep into the lungs and even enter the bloodstream, causing respiratory and cardiovascular issues.
            - **Environmental Impact**: Reduces visibility and contributes to haze.
            """)
    st.markdown("###  Did You Know?")
    st.info(
        "PM2.5 particles are approximately 30 times smaller than the diameter of a human hair!"
    )

    try:
        df = pd.read_csv("merged_data.csv")
        st.markdown("### Dataset Features")
        st.write(df.columns.tolist())
        st.markdown("### Dataset Preview")
        st.dataframe(df.head())
        st.markdown("### Download the Dataset")
        st.download_button(
            label="Download merged_data.csv",
            data=df.to_csv(index=False),
            file_name="merged_data.csv",
            mime="text/csv"
        )
    except FileNotFoundError:
        st.error("Error: 'merged_data.csv' file not found!")

# ====================== PAGE 2: EDA ======================
def page_eda():
    add_custom_css()
    st.title("Exploratory Data Analysis")
    st.markdown("## 📊 EDA Visualizations")

    st.markdown("### Dataset Overview")
    try:
        df = pd.read_csv("merged_data.csv")
    except Exception:
        st.error("Could not read merged_data.csv for EDA!")
        return

    st.write(df.head())

    st.markdown("### Missing Values Summary")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    # Missing Values Bar Plot
    st.markdown("### Missing Values Bar Plot")
    mv = missing_values[missing_values > 0]
    if not mv.empty:
        plt.figure(figsize=(10, 4))
        mv.sort_values(ascending=False).plot(kind='bar', color='skyblue')
        plt.title('Missing Values per Feature')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf()); plt.clf()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Key Features", fontsize=16)
    st.pyplot(plt.gcf()); plt.clf()

    # Correlation (absolute) with PM2.5
    st.markdown("### Feature Correlation with PM2.5")
    pm25_corr = corr_matrix['PM2.5'].drop('PM2.5').abs().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    sns.barplot(x=pm25_corr.values, y=pm25_corr.index, color="mediumseagreen")
    plt.title("Feature Correlation (abs) to PM2.5")
    plt.xlabel('Absolute Correlation')
    st.pyplot(plt.gcf()); plt.clf()

    # Time-series analysis for PM2.5
    st.markdown("### Time-Series Analysis of PM2.5 Concentration")
    if set(['year', 'month', 'day', 'hour']).issubset(df.columns):
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.set_index('datetime')
        daily_avg_pm25 = df['PM2.5'].resample('D').mean()
        plt.figure(figsize=(15, 6))
        plt.plot(daily_avg_pm25, label='Daily Average PM2.5')
        plt.title("Daily PM2.5 Concentration Over Time", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
        plt.legend()
        st.pyplot(plt.gcf()); plt.clf()

        # Monthly bar plot
        st.markdown("### Monthly PM2.5 Trend")
        monthly_avg_pm25 = df.groupby('month')['PM2.5'].mean()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=monthly_avg_pm25.index, y=monthly_avg_pm25.values, color="steelblue")
        plt.title("Average PM2.5 Concentration by Month", fontsize=16)
        plt.xlabel("Month", fontsize=14)
        plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
        st.pyplot(plt.gcf()); plt.clf()

    # Distribution
    st.markdown("### PM2.5 Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PM2.5'], bins=50, kde=True, color="purple")
    plt.title("Distribution of PM2.5 Concentration", fontsize=16)
    plt.xlabel("PM2.5 Concentration (µg/m³)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    st.pyplot(plt.gcf()); plt.clf()

    # Violin plot for station, if available
    if 'station' in df.columns:
        st.markdown("### PM2.5 Distribution by Station (Violin Plot)")
        plt.figure(figsize=(14,7))
        _station_order = df.groupby('station')['PM2.5'].mean().sort_values().index
        sns.violinplot(x='station', y='PM2.5', data=df.reset_index(), order=_station_order)
        plt.xticks(rotation=45)
        plt.title('PM2.5 by Station')
        st.pyplot(plt.gcf()); plt.clf()

    # PM2.5 vs Temperature
    if 'TEMP' in df.columns:
        st.markdown("### PM2.5 vs Temperature")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df['TEMP'], y=df['PM2.5'], alpha=0.6, color="orange")
        plt.title("Scatter Plot: PM2.5 vs Temperature", fontsize=16)
        plt.xlabel("Temperature (°C)", fontsize=14)
        plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
        st.pyplot(plt.gcf()); plt.clf()

    # PM2.5 vs Wind Speed
    if 'WSPM' in df.columns:
        st.markdown("### PM2.5 vs Wind Speed")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df['WSPM'], y=df['PM2.5'], alpha=0.6, color="green")
        plt.title("Scatter Plot: PM2.5 vs Wind Speed", fontsize=16)
        plt.xlabel("Wind Speed (m/s)", fontsize=14)
        plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
        st.pyplot(plt.gcf()); plt.clf()

    # Pairplot for key relations (limit size for speed)
    st.markdown("### Pairwise Relationships (Pairplot)")
    pairplot_cols = ['PM2.5','PM10','TEMP','PRES','WSPM']
    subset_df = df[pairplot_cols].dropna().sample(n=800, random_state=1) if df.shape[0]>800 else df[pairplot_cols].dropna()
    sns.pairplot(subset_df, diag_kind='kde', plot_kws={'alpha':0.5,'s':15})
    st.pyplot(plt.gcf()); plt.clf()

    st.markdown(
        "Visualizations above help in identifying distribution, trends, outliers, seasonal variation, spatial variation, and key linear associations for PM2.5 and major predictors."
    )

# ====================== PAGE 3: MODELING & PREDICTION ======================
def page_modeling_prediction():
    add_custom_css()
    st.title("Model Training & Evaluation")

    try:
        df = pd.read_csv("merged_data.csv")
    except Exception:
        st.error("Could not read merged_data.csv for Modeling & Prediction!")
        return

    st.markdown("#### Dataset Overview")
    st.dataframe(df.head())

    # Data Preparation
    FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    TARGET = "PM2.5"
    X = df[FEATURES].fillna(df[FEATURES].mean())
    y = df[TARGET].fillna(df[TARGET].mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define models
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "XGBoost Regressor": xgb.XGBRegressor(),
        "Ridge Regression": Ridge()
    }

    st.markdown("### Model Comparison")

    # Train and Evaluate All Models
    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({"Model": name,
                        "MSE": mse,
                        "R²": r2,
                        "Accuracy (%)": r2*100})
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values('Accuracy (%)', ascending=False)
    st.dataframe(results_df.style.format({'MSE': "{:.2f}", 'R²': "{:.3f}", 'Accuracy (%)': "{:.2f}"}), use_container_width=True)

    st.markdown("### Select a Model for Detailed Output")
    model_name = st.selectbox(
        "Choose a model for details:", 
        options=results_df["Model"].tolist(),
        label_visibility="visible"
    )
    model = trained_models[model_name]
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    accuracy = r2*100

    st.markdown(f"#### Results for **{model_name}**")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.3f}")
    st.write(f"**Model Accuracy:** {accuracy:.2f}%")

    # Feature importance plot (if available)
    if hasattr(model, "feature_importances_"):
        st.markdown("### Feature Importance")
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance", ascending=False)
        st.dataframe(fi_df)
        plt.figure(figsize=(7, 5))
        sns.barplot(y=fi_df["Feature"], x=fi_df["Importance"], color="royalblue")
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        st.pyplot(plt.gcf()); plt.clf()
    elif hasattr(model, "coef_"):
        coeff = model.coef_
        fi_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": coeff}).sort_values("Coefficient", key=abs, ascending=False)
        st.markdown("### Model Coefficients")
        st.dataframe(fi_df)
        plt.figure(figsize=(7, 5))
        sns.barplot(y=fi_df["Feature"], x=fi_df["Coefficient"], color="tomato")
        plt.title("Model Coefficients")
        plt.xlabel("Coefficient")
        plt.tight_layout()
        st.pyplot(plt.gcf()); plt.clf()

    st.markdown("___")
    st.info("Model summary completed. Visualize more details in the Prediction and Visualization section.")

# ====================== PAGE 4: PREDICTION & VISUALIZATION ======================
def page_prediction_and_viz():
    add_custom_css()
    st.title("Air Quality Prediction & Visualization")

    try:
        df = pd.read_csv("merged_data.csv")
    except Exception:
        st.error("Could not read merged_data.csv for Prediction & Visualization!")
        return

    st.markdown("#### Dataset Preview")
    st.dataframe(df.head())

    feature_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    target_col = "PM2.5"

    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
        comparison_data.append({"Model": name, "MSE": round(mse,2), "R²": round(r2,2), "Accuracy (%)": round(r2 * 100,2)})
    st.markdown("### 🔍 Model Comparison Overview")
    st.dataframe(pd.DataFrame(comparison_data).sort_values(by="Accuracy (%)", ascending=False), use_container_width=True)

    model_name = st.selectbox(
        "Choose a model for detailed visualization", 
        list(models.keys()),
        label_visibility="collapsed"  # Hides label but still provides one
    )
    model = models[model_name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.markdown("### Correlation Heatmap")
    corr = df[feature_cols + [target_col]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(plt.gcf()); plt.clf()

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    accuracy = r2 * 100
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    st.write(f"**Model Accuracy:** {accuracy:.2f}%")

    st.markdown("### Sample Predictions (First 10)")
    results_df = pd.DataFrame({"Actual PM2.5": y_test[:10].values, "Predicted PM2.5": predictions[:10]}).reset_index(drop=True)
    st.dataframe(results_df)

    st.markdown("### Visualization: Actual vs Predicted Scatter")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color="blue", label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect Prediction")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.legend()
    st.pyplot(plt.gcf()); plt.clf()

    st.markdown("### Visualization: Actual vs Predicted Line Plot")
    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:50].values, label="Actual", marker='o')
    plt.plot(predictions[:50], label="Predicted", marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("PM2.5 Value")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf()); plt.clf()

    st.markdown("### Model Comparison: Accuracy Bar Chart")
    model_names = [entry["Model"] for entry in comparison_data]
    accuracies = [entry["Accuracy (%)"] for entry in comparison_data]
    plt.figure(figsize=(10, 6))
    plt.barh(model_names, accuracies, color='green')
    plt.xlabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    st.pyplot(plt.gcf()); plt.clf()

    st.markdown("### Feature Distribution")
    for feature in feature_cols:
        st.markdown(f"#### Distribution of {feature}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30, color='purple')
        plt.title(f"Distribution of {feature}")
        st.pyplot(plt.gcf()); plt.clf()

    st.markdown("### Predicted vs Actual PM2.5 Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test, kde=True, color='blue', label='Actual PM2.5')
    sns.histplot(predictions, kde=True, color='red', label='Predicted PM2.5')
    plt.legend()
    plt.title("Distribution of Actual vs Predicted PM2.5")
    st.pyplot(plt.gcf()); plt.clf()

    st.markdown("### Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score", color="blue")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation Score", color="red")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Learning Curve")
    st.pyplot(plt.gcf()); plt.clf()

# ====================== PAGE 5: INTERACTIVE PREDICTION DASHBOARD ======================
def page_interactive_dashboard():
    add_custom_css()
    st.title("Interactive Prediction Tool")

    try:
        df = pd.read_csv("merged_data.csv")
    except Exception:
        st.error("Could not read merged_data.csv for Dashboard!")
        return

    feature_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    target_col = "PM2.5"

    X = df[feature_cols]
    y = df[target_col]
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model_name = st.selectbox(
        "Choose a regression model",
        [
            "Gradient Boosting Regressor",
            "Random Forest Regressor",
            "XGBoost Regressor",
            "Linear Regression",
            "Ridge Regression"
        ],
        label_visibility="visible"
    )

    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "XGBoost Regressor": xgb.XGBRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge()
    }
    model = models[model_name]
    model.fit(X_train, y_train)

    st.markdown("### Enter Feature Values")
    user_input = {}
    for feature in feature_cols:
        user_input[feature] = st.number_input(
            f"Value for {feature}:", 
            value=float(X[feature].mean()), 
            format="%.4f", 
            key=f"user_input_{feature}"
        )

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

# ====================== NAVIGATION / MAIN APP ======================
def main():
    st.set_page_config(page_title="Beijing Air Quality Analysis", layout="wide")
    add_custom_css()
    st.sidebar.markdown("## Navigation")
    pages = {
        "Introduction": page_introduction,
        "EDA": page_eda,
        "Modeling and Prediction": page_modeling_prediction,
        "Prediction and Visualization": page_prediction_and_viz,
        "Interactive Prediction Dashboard": page_interactive_dashboard
    }
    choice = st.sidebar.radio("Select a page", list(pages.keys()), label_visibility="collapsed")
    pages[choice]()

if __name__ == "__main__":
    main()