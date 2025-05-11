import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def add_custom_css():
    # CSS for black background
    st.markdown(
        """
        <style>
        /* Set the background to black */
        body {
            background-color: black;
            color: white; /* Ensure text is visible on a black background */
        }

        /* Modify headers and links for better visibility */
        h1, h2, h3, h4, h5, h6 {
            color: #00FF00; /* Green text for headers */
        }

        a {
            color: #1E90FF; /* Blue text for links */
        }

        /* Info and button sections styling */
        .st-info {
            background-color: #333333; /* Dark gray background for info boxes */
            color: white;
        }

        button {
            background-color: #444444; /* Dark gray buttons */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def run():
    # Add custom CSS
    add_custom_css()
    st.title("Exploratory Data Analysis")
    st.markdown("## 📊 EDA Visualizations")

    # Load data
    st.markdown("### Dataset Overview")
    df = pd.read_csv("merged_data.csv")

    st.write("Here is a preview of the dataset:")
    st.dataframe(df.head())

    # Handling missing values for EDA
    st.markdown("### Missing Values Summary")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    # Fill missing values for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Key Features", fontsize=16)
    st.pyplot(plt)

    # Time-series analysis for PM2.5
    st.markdown("### Time-Series Analysis of PM2.5 Concentration")
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    daily_avg_pm25 = df['PM2.5'].resample('D').mean()
    plt.figure(figsize=(15, 6))
    plt.plot(daily_avg_pm25, label='Daily Average PM2.5')
    plt.title("Daily PM2.5 Concentration Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
    plt.legend()
    st.pyplot(plt)

    # Monthly analysis of PM2.5
    st.markdown("### Monthly PM2.5 Trend")
    df['month'] = df.index.month
    monthly_avg_pm25 = df.groupby('month')['PM2.5'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_avg_pm25.index, y=monthly_avg_pm25.values, palette="Blues_d")
    plt.title("Average PM2.5 Concentration by Month", fontsize=16)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
    st.pyplot(plt)

    # PM2.5 distribution
    st.markdown("### PM2.5 Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PM2.5'], bins=50, kde=True, color="purple")
    plt.title("Distribution of PM2.5 Concentration", fontsize=16)
    plt.xlabel("PM2.5 Concentration (µg/m³)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    st.pyplot(plt)

    # PM2.5 vs temperature
    st.markdown("### PM2.5 vs Temperature")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['TEMP'], y=df['PM2.5'], alpha=0.6, color="orange")
    plt.title("Scatter Plot: PM2.5 vs Temperature", fontsize=16)
    plt.xlabel("Temperature (°C)", fontsize=14)
    plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
    st.pyplot(plt)

    # PM2.5 vs wind speed
    st.markdown("### PM2.5 vs Wind Speed")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['WSPM'], y=df['PM2.5'], alpha=0.6, color="green")
    plt.title("Scatter Plot: PM2.5 vs Wind Speed", fontsize=16)
    plt.xlabel("Wind Speed (m/s)", fontsize=14)
    plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
    st.pyplot(plt)

    st.markdown(
        "Through these visualizations, we can observe patterns and relationships between PM2.5 concentration and other environmental variables."
    )

if __name__ == "__main__":
    run()
