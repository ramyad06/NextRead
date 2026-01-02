from fastapi import FastAPI, HTTPException
import pandas as pd
import sys
import os
import uvicorn
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.tfidf_recommender import TFIDFRecommender
from models.xgboost_ranker import XGBRanker
from models.popularity_recommender import PopularityRecommender
from main import prepare_candidate_features # Reuse helper

# Globals
articles_df = None
interactions_df = None
users_df = None
tfidf_rec = None
xgb_ranker = None
pop_rec = None
url_to_idx = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Data and Train Models on Startup
    global articles_df, interactions_df, users_df
    global tfidf_rec, xgb_ranker, pop_rec, url_to_idx
    
    print("Loading data...")
    articles_df = pd.read_csv('data/raw/articles.csv')
    interactions_df = pd.read_csv('data/processed/interactions.csv')
    users_df = pd.read_csv('data/processed/users.csv')
    
    url_to_idx = {url: idx for idx, url in enumerate(articles_df['url'])}
    
    print("Training TF-IDF Recommender...")
    tfidf_rec = TFIDFRecommender()
    tfidf_rec.fit(articles_df)
    
    print("Training XGBoost Ranker...")
    xgb_ranker = XGBRanker()
    # Using all data for training in this demo service
    X, y, groups = xgb_ranker.prepare_data(interactions_df, articles_df, users_df, tfidf_rec)
    xgb_ranker.tune_hyperparameters(X, y, groups)
    xgb_ranker.train(X, y, groups)
    
    print("Training Popularity Recommender...")
    pop_rec = PopularityRecommender()
    pop_rec.fit(interactions_df, articles_df)
    
    print("System Ready!")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = 5):
    # Check if user exists in our "db"
    if user_id not in users_df['user_id'].values:
        # Pure Cold Start for completely new user ID
        # In a real system we might track anonymous users differently
        print(f"New user {user_id}, using Popularity Fallback")
        recs = pop_rec.recommend(top_k=top_k)
        return {"user_id": user_id, "strategy": "popularity", "recommendations": recs}
        
    # Get User History
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    
    if len(user_interactions) < 2:
         # Cold Start Fallback
         print(f"User {user_id} has little history, using Popularity Fallback")
         recs = pop_rec.recommend(top_k=top_k)
         return {"user_id": user_id, "strategy": "popularity", "recommendations": recs}

    history_indices = [url_to_idx.get(u) for u in user_interactions['article_url'] if u in url_to_idx]
    
    if not history_indices:
        recs = pop_rec.recommend(top_k=top_k)
        return {"user_id": user_id, "strategy": "popularity", "recommendations": recs}

    # Candidate Gen
    user_profile = tfidf_rec.get_user_profile(history_indices)
    candidates = tfidf_rec.get_candidates(user_profile, top_k=20, exclude_indices=history_indices)
    
    if not candidates:
        recs = pop_rec.recommend(top_k=top_k)
        return {"user_id": user_id, "strategy": "popularity", "recommendations": recs}
        
    # Feature Prep for Ranking
    user_stats = user_interactions.agg({
        'read_time_seconds': 'mean',
        'clicked': 'count'
    })
    user_avg_read_time = user_stats['read_time_seconds'] if not pd.isna(user_stats['read_time_seconds']) else 60
    user_total_clicks = user_stats['clicked'] if not pd.isna(user_stats['clicked']) else 0
    
    cand_list = []
    for cand in candidates:
         cand_list.append(prepare_candidate_features(cand, interactions_df, articles_df, user_avg_read_time, user_total_clicks))
    
    cand_df = pd.DataFrame(cand_list)
    
    if cand_df.empty:
        recs = pop_rec.recommend(top_k=top_k)
        return {"user_id": user_id, "strategy": "popularity", "recommendations": recs}
        
    # Predict
    # Rename feature if needed (fixed in main.py already)
    # Ensure columns match those expected by ranker
    # The prepare_candidate_features in main.py now returns 'similarity_score'
    
    scores = xgb_ranker.predict(cand_df[xgb_ranker.feature_cols])
    cand_df['rerank_score'] = scores
    ranked_df = cand_df.sort_values(by='rerank_score', ascending=False)
    
    top_recs = ranked_df.head(top_k)
    
    response = []
    for _, row in top_recs.iterrows():
        response.append({
            "url": row['url'],
            "title": row['title'],
            "score": float(row['rerank_score'])
        })
        
    return {"user_id": user_id, "strategy": "hybrid", "recommendations": response}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
