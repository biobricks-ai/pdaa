import streamlit as st
from app_heatmaps import run_heatmap_app
from app_dart import run_dart_app
# from app_finetuning import run_finetuning_app

st.set_page_config(page_title="Multi-Tool App", layout="wide")

# Create tabs on the left (collapsible) 
tab_names = ["Heatmap Tool", "Dart Tool"]
selected_tab = st.sidebar.radio("Choose a tool:", tab_names)

if selected_tab == "Heatmap Tool":
    run_heatmap_app()

elif selected_tab == "Dart Tool":
    run_dart_app()


# # Create top navigation using tabs, each tab loads different app.
# tabs = st.tabs(["Heatmap Tool", "Dart Tool"])

# with tabs[0]:  
#     run_heatmap_app()

# with tabs[1]:
#     run_dart_app()