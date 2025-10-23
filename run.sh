#!/bin/bash

echo "Setting up environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running application..."
streamlit run app.py