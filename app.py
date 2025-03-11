import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Define location data with coordinates, file paths, and region types
location_data = {
    "Data_Wanliu": (39.974, 116.311, "path/to/Wanliu.csv", "Urban"),
    "Data_Wanshouxigong": (39.889, 116.352, "path/to/Wanshouxigong.csv", "Urban"),
    "Data_Aotizhongxin": (40.007, 116.397, "path/to/Aotizhongxin.csv", "Urban"),
    "Data_Changping": (40.22, 116.23, "path/to/Changping.csv", "Rural"),
    "Data_Dingling": (40.29, 116.22, "path/to/Dingling.csv", "Hills"),
    "Data_Dongsi": (39.93, 116.42, "path/to/Dongsi.csv", "Urban"),
    "Data_Guanyuan": (39.93, 116.36, "path/to/Guanyuan.csv", "Urban"),
    "Data_Gucheng": (39.91, 116.18, "path/to/Gucheng.csv", "Urban"),
    "Data_Huairou": (40.31, 116.63, "path/to/Huairou.csv", "Forest"),
    "Data_Nongzhanguan": (39.93, 116.47, "path/to/Nongzhanguan.csv", "Urban"),
    "Data_Shunyi": (40.13, 116.65, "path/to/Shunyi.csv", "Rural"),
    "Data_Tiantan": (39.88, 116.41, "path/to/Tiantan.csv", "Urban"),
    "Data_Bohai": (38.88, 118.15, "path/to/Bohai.csv", "Sea")
}

# Streamlit page configuration
st.set_page_config(page_title="Air Quality Data EDA", layout="wide")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Main Analysis", "Merged Data EDA", "Weather Forecasting"] + list(location_data.keys()))
visualization_options = ["Histogram", "Box Plot", "Scatter Plot", "Time Series", "Heatmap"]
selected_visualization = st.sidebar.multiselect("Choose visualizations", visualization_options)

def perform_eda(data):
    st.subheader("Statistical Summary")
    st.write(data.describe())
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    
    if "Heatmap" in selected_visualization:
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

def train_lstm_model(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['PM2.5']].dropna())
    X, y = [], []
    
    for i in range(len(scaled_data) - 10):
        X.append(scaled_data[i:i+10])
        y.append(scaled_data[i+10])
    
    X, y = np.array(X), np.array(y)
    
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(10, 1)),
        keras.layers.LSTM(50),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    return model, scaler

def predict_future(model, scaler, data):
    last_10_days = scaler.transform(data[['PM2.5']].dropna())[-10:]
    future_input = np.array([last_10_days])
    predicted_value = model.predict(future_input)
    return scaler.inverse_transform(predicted_value)[0][0]

if page == "Main Analysis":
    st.title("Air Quality Data EDA")
    total_locations = st.number_input("Enter number of locations to analyze:", min_value=1, max_value=len(location_data), value=2, step=1)
    selected_locations = st.multiselect("Choose locations for EDA:", list(location_data.keys()), default=list(location_data.keys())[:total_locations])
    
    if selected_locations:
        merged_data = pd.DataFrame()
        
        for location in selected_locations:
            lat, lon, file_path, region_type = location_data[location]
            try:
                data = pd.read_csv(file_path)
                data['Location'] = location
                data['Region Type'] = region_type
                merged_data = pd.concat([merged_data, data], ignore_index=True)
            except Exception as e:
                st.write(f"Error reading the file for {location}: {e}")
        
        st.subheader("Geographical Map of Selected Locations")
        map_center = [39.9, 116.4]
        air_quality_map = folium.Map(location=map_center, zoom_start=10)
        
        for location in selected_locations:
            lat, lon, _, region_type = location_data[location]
            aqi = merged_data[merged_data['Location'] == location]['PM2.5'].mean()
            popup_text = f"<b>{location}</b><br>Region Type: {region_type}<br>Avg AQI (PM2.5): {aqi:.2f}"
            folium.Marker(
                location=[lat, lon], 
                popup=popup_text, 
                icon=folium.Icon(color="blue" if region_type in ["Urban", "Sea"] else "green")
            ).add_to(air_quality_map)
        
        folium_static(air_quality_map)
    else:
        st.write("Please select at least one location to perform EDA.")

elif page == "Merged Data EDA":
    st.title("Merged Data EDA")
    if 'merged_data' in locals():
        perform_eda(merged_data)
    else:
        st.write("No data available. Please select locations in 'Main Analysis'.")

elif page == "Weather Forecasting":
    st.title("Weather Forecasting using LSTM")
    selected_location = st.selectbox("Choose a location for forecasting:", list(location_data.keys()))
    lat, lon, file_path, region_type = location_data[selected_location]
    data = pd.read_csv(file_path)
    model, scaler = train_lstm_model(data)
    forecast_value = predict_future(model, scaler, data)
    st.subheader(f"Predicted PM2.5 value for the next day: {forecast_value:.2f}")

else:
    st.title(f"EDA for {page}")
    lat, lon, file_path, region_type = location_data[page]
    data = pd.read_csv(file_path)
    perform_eda(data)
