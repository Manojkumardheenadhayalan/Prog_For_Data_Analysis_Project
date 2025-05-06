import streamlit as st
import importlib

# Set page configuration
st.set_page_config(page_title="Beijing Air Quality Analysis", layout="wide")

def set_black_background_and_sidebar():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: black;
            color: white;
        }

        [data-testid="stSidebarContent"], [data-testid="stHeader"] {
            background-color: black;
            color: white;
        }

        .stButton > button {
            background-color: #333333;
            color: white;
        }

        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: white;
        }

        .stDataFrame, .stTable {
            background-color: #222222;
            color: white;
        }

        [role="radiogroup"] > div {
            background-color: #222222;
            margin-bottom: 5px;
            border-radius: 10px;
            padding: 5px;
        }

        [role="radiogroup"] label {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_black_background_and_sidebar()

    st.sidebar.markdown("## Navigation")
    pages = {
        "Introduction": "Introduction",
        "EDA": "EDA",
        "Modeling and Prediction": "Modeling_and_Prediction",
        "Prediction and Visualization": "Prediction_and_Visualization",
        "Interactive Prediction Dashboard": "Interactive_Prediction_Dashboard"
    }

    choice = st.sidebar.radio("", list(pages.keys()))
    module_name = pages[choice]

    try:
        page_module = importlib.import_module(f"custom_pages.{module_name}")
        page_module.run()
    except ModuleNotFoundError:
        st.error(f"Page module 'custom_pages/{module_name}.py' not found.")
    except AttributeError:
        st.error(f"The module 'custom_pages/{module_name}.py' must have a `run()` function.")

if __name__ == "__main__":
    main()
