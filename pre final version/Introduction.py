import streamlit as st
import pandas as pd

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
    # Page Title
    st.title("Introduction")
    
    # Project Overview
    st.markdown(
        """
        ##  Project Overview
        This project aims to predict the **PM2.5 concentration** in Beijing from 2013 to 2017 using advanced machine learning techniques. 
        Understanding PM2.5 levels is crucial for environmental planning and improving air quality.
        """
    )
    
    # Dataset Information
    st.markdown(
        """
        ## Dataset Information
        - **Source**: Beijing Environmental Monitoring Center
        - **Time Period**: March 1, 2013, to February 28, 2017
        - **Features**: 
            - Air pollutant levels (PM2.5, PM10, SO2, NO2, CO, O3)
            - Meteorological data (Temperature, Pressure, Wind Speed, Rainfall)
        - **Sites**: Data from **12 monitoring stations** across Beijing.
        
        Missing values in the dataset are denoted as `NA` and have been handled appropriately during preprocessing.
        """
    )

    # Key Objectives Section
    st.markdown(
        """
        ##  Key Objectives
        - Perform **exploratory data analysis (EDA)** to gain insights into PM2.5 trends and their correlation with other factors.
        - Build and evaluate **regression models** to predict PM2.5 concentrations.
        - Visualize results for better understanding and actionable insights.
        """
    )

    # Interactive Section: Why is PM2.5 Important?
    st.markdown("###  Why is PM2.5 Important?")
    if st.button("Learn More"):
        st.write(
            """
            PM2.5 particles are fine particulate matter with a diameter of less than 2.5 micrometers.
            - **Health Impact**: Can penetrate deep into the lungs and even enter the bloodstream, causing respiratory and cardiovascular issues.
            - **Environmental Impact**: Reduces visibility and contributes to haze.
            """
        )

    # Fun Fact Section
    st.markdown("###  Did You Know?")
    st.info(
        "PM2.5 particles are approximately 30 times smaller than the diameter of a human hair!"
    )

    # Load the merged data
    try:
        df = pd.read_csv("merged_data.csv")
        # Display dataset features
        st.markdown("### Dataset Features")
        st.write(df.columns.tolist())  # Show column names

        # Display dataset preview
        st.markdown("### Dataset Preview")
        st.dataframe(df.head())  # Show the first few rows of the dataset

        # Allow the user to download the dataset
        st.markdown("### Download the Dataset")
        st.download_button(
            label="Download merged_data.csv",
            data=df.to_csv(index=False),
            file_name="merged_data.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("Error: 'merged_data.csv' file not found!")

if __name__ == "__main__":
    run()
