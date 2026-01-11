#!/bin/bash
# ============================================
# Geo-AI Feature Extraction App Launcher
# ============================================

# Change to app directory
cd "$(dirname "$0")"

# Activate virtual environment
source ../venv/bin/activate

# Run Streamlit
echo "🌍 Starting Geo-AI Feature Extraction..."
echo "================================================"
streamlit run app.py --server.headless true --server.port 8501

# Deactivate on exit
deactivate
