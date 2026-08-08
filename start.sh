#!/bin/bash

# Activate pdaa environment
conda activate pdaa
# Starting blaze graph
java -server -Xmx4g -jar blazegraph.jar 
# java -server -Xmx4g -jar blazegraph.jar &

# Wait for Blazegraph to initialize (adjust as needed)
sleep 10

# Running app
streamlit run app.py --server.port 8501 # --server.address 0.0.0.0

python -m streamlit run app.py --server.port 8501

