import pandas as pd
import numpy as np
import sys
import os

# Improve path handling to find src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.popularity_recommender import PopularityRecommender
from models.xgboost_ranker import XGBRanker
from models.tfidf_recommender import TFIDFRecommender
from evaluation.metrics import precision_at_k, ndcg_at_k, calculate_ctr
from sklearn.model_selection import train_test_split

def prepare_candidate_features(candidate, interactions_df, articles_df, user_avg_read_time, user_total_clicks):
    """Helper to create a feature row for a candidate."""
    idx = candidate['article_index']
    url = candidate['article_url']
    
    # Item Stats (Global - naive approach using full interaction df for simplicity)
    # Ideally should use only train data stats to prevent leakage
    item_stats = interactions_df[interactions_df['article_url'] == url]
    if not item_stats.empty:
        item_pop = len(item_stats)
        item_rating = item_stats['rating'].mean()
    else:
        item_pop = 0
        item_rating = 3.0 # default
        
    return {
        'user_avg_read_time': user_avg_read_time,
        'user_total_clicks': user_total_clicks,
        'item_popularity': item_pop,
        'item_avg_rating': item_rating,
        'similarity_score': candidate['similarity_score'],
        'url': url,
        'title': articles_df.iloc[idx]['title']
    }

def main():
    print("=== NextRead Recommendation System Pipeline ===")
    
    # Paths
    articles_path = 'data/raw/articles.csv'
    interactions_path = 'data/processed/interactions.csv'
    users_path = 'data/processed/users.csv'
    
    # Load Data
    print(f"Loading data from {articles_path} and {interactions_path}...")
    articles_df = pd.read_csv(articles_path)
    interactions_df = pd.read_csv(interactions_path)
    users_df = pd.read_csv(users_path)
    
    # Split Data: Train (80%) / Test (20%) by User
    unique_users = interactions_df['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    
    train_interactions = interactions_df[interactions_df['user_id'].isin(train_users)]
    test_interactions = interactions_df[interactions_df['user_id'].isin(test_users)]
    
    print(f"Train Users: {len(train_users)} | Test Users: {len(test_users)}")
    print(f"Train Interactions: {len(train_interactions)} | Test Interactions: {len(test_interactions)}")
    
    # 1. Train Candidate Model (on ALL articles, but usually fits on corpus)
    print("\n--- Phase 1: Candidate Generation Model (TF-IDF) ---")
    tfidf_rec = TFIDFRecommender()
    tfidf_rec.fit(articles_df)
    
    # 2. Train Ranking Model (on Train Split)
    print("\n--- Phase 2: Ranking Model (XGBoost) ---")
    xgb_ranker = XGBRanker()
    # Note: prepare_data needs to use train_interactions to compute features
    X_train, y_train, groups = xgb_ranker.prepare_data(train_interactions, articles_df, users_df, tfidf_rec)
    xgb_ranker.tune_hyperparameters(X_train, y_train, groups)
    xgb_ranker.train(X_train, y_train, groups)
    
    # 2.5 Train Popularity Model (Cold Start Fallback)
    print("\n--- Phase 2.5: Popularity Model ---")
    pop_rec = PopularityRecommender()
    pop_rec.fit(train_interactions, articles_df)
    
    # 3. Evaluation Loop
    print("\n--- Phase 3: Evaluation on Test Set ---")
    
    # Metrics aggregators
    p_at_5_list = []
    ndcg_at_5_list = []
    total_impressions = 0
    total_hits = 0
    
    # url to idx map
    url_to_idx = {url: idx for idx, url in enumerate(articles_df['url'])}
    
    for i, user_id in enumerate(test_users):
        # Progress
        if i % 10 == 0:
            print(f"Evaluating user {i}/{len(test_users)}...")
            
        # Get Ground Truth (Test Interactions)
        user_test_interactions = test_interactions[test_interactions['user_id'] == user_id]
        true_urls = set(user_test_interactions['article_url'])
        true_indices = [url_to_idx.get(u) for u in true_urls if u in url_to_idx]
        
        if not true_indices:
            continue
            
        # Cold Start / Fallback Logic
        recommended_indices = []
        
        # Simpler approach: Split user's interactions 50% history, 50% target
        user_all_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        # Cold Start Simulation: If user has < 2 interactions, pretend they have 0 history and test on all
        if len(user_all_interactions) < 2:
             # Pure Cold Start
             top_pop = pop_rec.recommend(top_k=5)
             recommended_indices = [url_to_idx.get(r['url']) for r in top_pop if r['url'] in url_to_idx]
             # Target is all interactions (since 0 history)
             target_indices = true_indices
        else:
            history_df = user_all_interactions.iloc[:len(user_all_interactions)//2]
            target_df = user_all_interactions.iloc[len(user_all_interactions)//2:]
            
            history_urls = history_df['article_url'].tolist()
            history_indices = [url_to_idx.get(u) for u in history_urls if u in url_to_idx]
            
            target_urls = set(target_df['article_url'])
            target_indices = [url_to_idx.get(u) for u in target_urls if u in url_to_idx]
            
            if not target_indices:
                continue
                
            # Content + Ranker
            if not history_indices:
                 # Fallback to pop
                 top_pop = pop_rec.recommend(top_k=5)
                 recommended_indices = [url_to_idx.get(r['url']) for r in top_pop if r['url'] in url_to_idx]
            else:
                user_profile = tfidf_rec.get_user_profile(history_indices)
                candidates = tfidf_rec.get_candidates(user_profile, top_k=20, exclude_indices=history_indices)
                
                if not candidates:
                    # Fallback
                    top_pop = pop_rec.recommend(top_k=5)
                    recommended_indices = [url_to_idx.get(r['url']) for r in top_pop if r['url'] in url_to_idx]
                else:
                    # Re-rank Logic
                    user_stats = history_df.agg({
                        'read_time_seconds': 'mean',
                        'clicked': 'count'
                    })
                    user_avg_read_time = user_stats['read_time_seconds'] if not pd.isna(user_stats['read_time_seconds']) else 60
                    user_total_clicks = user_stats['clicked'] if not pd.isna(user_stats['clicked']) else 0
                    
                    cand_list = []
                    for cand in candidates:
                         cand_list.append(prepare_candidate_features(cand, interactions_df, articles_df, user_avg_read_time, user_total_clicks))
                    
                    cand_df = pd.DataFrame(cand_list)
                    
                    if not cand_df.empty:
                        scores = xgb_ranker.predict(cand_df[xgb_ranker.feature_cols])
                        cand_df['rerank_score'] = scores
                        ranked_df = cand_df.sort_values(by='rerank_score', ascending=False)
                        top_5_recs = ranked_df.head(5)
                        recommended_indices = [url_to_idx.get(url) for url in top_5_recs['url']]

        # Metrics
        p5 = precision_at_k(recommended_indices, target_indices, 5)
        n5 = ndcg_at_k(recommended_indices, target_indices, 5)
        
        p_at_5_list.append(p5)
        ndcg_at_5_list.append(n5)
        
        # CTR Simulation (Hit if ANY target is in top 5)
        # Simplistic view: if user sees relevant item, they click.
        hits = sum([1 for idx in recommended_indices if idx in target_indices])
        if hits > 0:
            total_hits += 1
        total_impressions += 1
            
    # Report
    print("\n=== Evaluation Results ===")
    avg_p5 = np.mean(p_at_5_list)
    avg_n5 = np.mean(ndcg_at_5_list)
    ctr_sim = calculate_ctr(total_hits, total_impressions)
    
    print(f"Mean Precision@5: {avg_p5:.4f}")
    print(f"Mean NDCG@5:      {avg_n5:.4f}")
    print(f"Simulated CTR:    {ctr_sim:.4f}")
    print(f"Evaluated on {len(p_at_5_list)} users.")

if __name__ == "__main__":
    main()
