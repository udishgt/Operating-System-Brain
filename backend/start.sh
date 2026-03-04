#!/bin/bash

echo ""
echo " OSB - Operating System Brain"
echo " =============================="
echo ""

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install deps
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Copy .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo " ACTION REQUIRED:"
    echo " Open .env and add your GEMINI_API_KEY"
    echo " Get free key at: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Press Enter after adding your key..."
fi

echo ""
echo " Starting OSB backend on http://localhost:8000"
echo " API docs: http://localhost:8000/docs"
echo " Press Ctrl+C to stop"
echo ""
python main.py
