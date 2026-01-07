#!/bin/bash

# Configuration
VENV_DIR="venv"
DATA_DIR="data"
RAW_DATA="$DATA_DIR/raw/articles.csv"
PROCESSED_DATA="$DATA_DIR/processed/interactions.csv"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "🚀 Starting NextRead Setup & Run..."

# 1. Environment Setup
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

echo "⬇️  Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# 2. Data Verification
if [ ! -f "$RAW_DATA" ] || [ ! -f "$PROCESSED_DATA" ]; then
    echo "⚠️  Data missing. Running scraper and generator..."
    # Create directories if they don't exist
    mkdir -p "$DATA_DIR/raw" "$DATA_DIR/processed"
    
    # Run data pipeline (Assuming scripts exist as per README)
    # Check if necessary directories exist, if not create them
    # Note: Scraper might take time, for now just warning if missing.
    # But user_generator.py is needed for interactions.
    
    if [ ! -f "$PROCESSED_DATA" ]; then
         echo "Generating synthetic user data..."
         python src/processing/user_generator.py
    fi
else
    echo "✅ Data found."
fi

# 3. Model Training (Offline)
echo "🧠 Running offline training pipeline..."
python main.py

# 4. Start Servers
echo "🌐 Starting Backend (Port 8000) and Frontend (Port 3000)..."

# Kill running processes on these ports if any (optional, be careful)
# lsof -ti:8000 | xargs kill -9 2>/dev/null
# lsof -ti:3000 | xargs kill -9 2>/dev/null

# Start Backend
uvicorn app:app --port 8000 --reload &
BACKEND_PID=$!

# Start Frontend
python3 -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

echo "✅ App is running!"
echo "➡️  Frontend: http://localhost:3000"
echo "➡️  Backend:  http://localhost:8000"
echo "Press CTRL+C to stop."

# Wait for user to exit
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT
wait
