import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import FactorAnalysis

# Import all the functions from your existing code
from processing_functions import (
    clean_gci_data, interpolate_gci_data, forecast_gci_data, scale_gci_data,
    clean_sdg_data, interpolate_sdg_data, forecast_sdg_data, scale_sdg_data,
    map_sdgs_to_variables, perform_factor_analysis, merge_and_map_data
)
from modeling_functions import SDGHRModeling

# Set page config
st.set_page_config(
    page_title="SDG and HR Metrics Analysis",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
        .main {padding: 2rem;}
        .stButton > button {width: 100%;}
        .metric-card {
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Function to create necessary directories
def create_directories():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

# Function to process data
@st.cache_data
def process_all_data():
    try:
        # Create directories
        create_directories()
        
        # Process GCI data
        gci_data = pd.read_excel('data/raw/WEF.xlsx')
        gci_processed = process_gci_data(gci_data)
        
        # Process SDG data
        sdg_data = pd.read_excel('data/raw/SDGs.xlsx')
        sdg_processed = process_sdg_data(sdg_data)
        
        # Merge and map data
        final_data, sdg_columns = merge_and_map_data(gci_processed, sdg_processed)
        
        return final_data, sdg_columns
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None

# Function to train models
@st.cache_resource
def train_models(data):
    modeling = SDGHRModeling(data)
    modeling.prepare_data()
    results = modeling.train_and_evaluate()
    feature_importance = modeling.calculate_feature_importance()
    return modeling, results, feature_importance

# Main app
def main():
    st.title("SDG and HR Metrics Analysis Dashboard")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        ["Data Processing", "Data Analysis", "Model Results", "Feature Importance"]
    )
    
    # Data Processing Page
    if page == "Data Processing":
        st.header("Data Processing")
        
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                final_data, sdg_columns = process_all_data()
                if final_data is not None:
                    st.success("Data processing completed!")
                    st.write("Final Data Shape:", final_data.shape)
                    st.write("SDG Mapping:")
                    st.write(sdg_columns)
        
        # Option to upload files
        st.subheader("Upload Data Files")
        gci_file = st.file_uploader("Upload WEF.xlsx", type="xlsx")
        sdg_file = st.file_uploader("Upload SDGs.xlsx", type="xlsx")
        
        if gci_file and sdg_file:
            if st.button("Process Uploaded Files"):
                with st.spinner("Processing uploaded files..."):
                    # Save uploaded files
                    with open(os.path.join("data/raw", "WEF.xlsx"), "wb") as f:
                        f.write(gci_file.getvalue())
                    with open(os.path.join("data/raw", "SDGs.xlsx"), "wb") as f:
                        f.write(sdg_file.getvalue())
                    
                    # Process data
                    final_data, sdg_columns = process_all_data()
                    if final_data is not None:
                        st.success("Uploaded files processed successfully!")
    
    # Data Analysis Page
    elif page == "Data Analysis":
        st.header("Data Analysis")
        
        try:
            data = pd.read_excel('data/processed/final_merged_data.xlsx')
            
            # Data overview
            st.subheader("Dataset Overview")
            st.write(data.head())
            
            # Time series analysis
            st.subheader("Time Series Analysis")
            metric = st.selectbox(
                "Select Metric",
                options=[col for col in data.columns if col not in ['Country', 'year', 'Country_year']]
            )
            
            fig = px.line(data, x='year', y=metric, color='Country')
            st.plotly_chart(fig)
            
            # Correlation analysis
            st.subheader("Correlation Analysis")
            correlation_matrix = data.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(correlation_matrix)
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error("Please process the data first.")
    
    # Model Results Page
    elif page == "Model Results":
        st.header("Model Results")
        
        try:
            data = pd.read_excel('data/processed/final_merged_data.xlsx')
            modeling, results, _ = train_models(data)
            
            st.subheader("Model Performance")
            st.write(results)
            
            # Display learning curves
            st.subheader("Learning Curves")
            model_files = os.listdir('plots')
            learning_curves = [f for f in model_files if f.startswith('learning_curve')]
            
            for curve in learning_curves:
                st.image(f'plots/{curve}')
                
        except Exception as e:
            st.error("Please process the data and train models first.")
    
    # Feature Importance Page
    elif page == "Feature Importance":
        st.header("Feature Importance Analysis")
        
        try:
            data = pd.read_excel('data/processed/final_merged_data.xlsx')
            _, _, feature_importance = train_models(data)
            
            # Display feature importance heatmap
            st.subheader("Feature Importance Heatmap")
            pivot_table = feature_importance.pivot(
                index='SDG',
                columns='Target',
                values='Importance'
            )
            fig = px.imshow(pivot_table,
                           title="Feature Importance Heatmap")
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error("Please process the data and train models first.")

if __name__ == "__main__":
    main()