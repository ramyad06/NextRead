# NextRead: Hybrid Article Recommendation System

NextRead is a production-ready hybrid recommendation system designed for newsletter platforms like Substack. It combines content-based filtering with Learning-to-Rank (LTR) techniques to deliver personalized article recommendations.

## 🚀 Features

*   **Hybrid Architecture**: Combines **TF-IDF** for candidate generation and **XGBoost** for learning-to-rank.
*   **Cold Start Handling**: Implements a **Popularity Fallback** ensures new users or users with sparse history still receive relevant recommendations.
*   **Real-time Inference**: Exposed via a high-performance **FastAPI** service.
*   **Automated Evaluation**: Includes offline metrics calculation (Precision@K, NDCG@K) and Simulated CTR.
*   **Synthetic Data Generation**: logic to generate realistic user interactions (clicks, read times) for training.
*   **Scalable Scraper**: Robust web scraper for Substack newsletters.

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **ML Libraries**: `scikit-learn`, `xgboost`, `pandas`, `numpy`
*   **API**: `fastapi`, `uvicorn`
*   **Data Processing**: `beautifulsoup4`, `requests`

## 📂 Project Structure

```
NextRead/
├── app.py                   # FastAPI application entry point
├── main.py                  # Offline training and evaluation pipeline
├── requirements.txt         # Project dependencies
├── data/
│   ├── raw/articles.csv           # Scraped articles
│   └── processed/interactions.csv # Synthetic user interactions
└── src/
    ├── scraping/            # Web scraping logic
    ├── processing/          # Data generation and processing
    ├── models/              # ML Models (TF-IDF, XGBoost, Popularity)
    └── evaluation/          # Metrics and Validation
```

## ⚡ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ramyad06/NextRead.git
    cd NextRead
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### 1. Data Collection & Generation
If you want to allow fresh data:
```bash
# Scrape articles (Configured for Lenny's, Pragmatic Engineer, etc.)
python src/scraping/scraper.py

# Generate synthetic user interactions
python src/processing/user_generator.py
```

### 2. Offline Training & Evaluation
Run the full pipeline to train models and view offline metrics (NDCG, Precision):
```bash
python main.py
```

### 3. Start the API Server
Start the FastAPI server for real-time inference:
```bash
uvicorn app:app --port 8000 --reload
```

## 🔌 API Documentation

### `GET /recommend/{user_id}`

Get personalized recommendations for a user.

**Parameters:**
*   `user_id` (int): Unique ID of the user.
*   `top_k` (int, optional): Number of articles to recommend. Default: 5.

**Example Request:**
```bash
# Get recommendations for User 0
curl "http://localhost:8000/recommend/0?top_k=3"
```

**Example Response:**
```json
{
  "user_id": 0,
  "strategy": "hybrid",
  "recommendations": [
    {
      "url": "https://www.lennysnewsletter.com/p/everyone-should-be-using-claude-code",
      "title": "Everyone should be using Claude Code more",
      "score": 1.019
    },
    ...
  ]
}
```

**Cold Start Example (New User):**
```bash
curl "http://localhost:8000/recommend/9999"
```
*Returns popularity-based results.*

## 📊 Performance

*   **Precision@5**: ~0.09
*   **NDCG@5**: ~0.09
*   **Simulated CTR**: ~35%
*(Metrics based on a synthetic dataset of ~50 articles and ~2000 interactions with negative sampling)*
