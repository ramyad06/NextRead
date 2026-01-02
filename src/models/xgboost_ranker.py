import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, GroupKFold

class XGBRanker:
    def __init__(self):
        self.model = xgb.XGBRanker(
            objective='rank:ndcg',
            learning_rate=0.1,
            gamma=1.0,
            min_child_weight=0.1,
            max_depth=6,
            n_estimators=100
        )
        self.feature_cols = []
        
    def prepare_data(self, interactions_df, articles_df, users_df, tfidf_recommender):
        """
        Prepares data for LTR.
        """
        print("Preparing data for Ranker...")
        # Merge data
        df = interactions_df.merge(articles_df, left_on='article_url', right_on='url', how='left')
        
        # Feature Engineering
        # 1. User Features
        user_stats = interactions_df.groupby('user_id').agg({
            'read_time_seconds': 'mean',
            'clicked': 'count'
        }).rename(columns={'read_time_seconds': 'user_avg_read_time', 'clicked': 'user_total_clicks'})
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        # 2. Item Features
        item_stats = interactions_df.groupby('article_url').agg({
            'clicked': 'count',
            'rating': 'mean'
        }).rename(columns={'clicked': 'item_popularity', 'rating': 'item_avg_rating'})
        
        df = df.merge(item_stats, left_on='url', right_index=True, how='left')
        
        # 3. Content Similarity Feature (Query-Document Feature)
        sim_dict = {}
        # Optimization: Group by user
        # Re-using previous optimization idea: Calculate on the fly or pre-calc
        # For speed in this demo, let's compute row-by-row but optimized via vectorization if possible
        # Or just loop over groups as before.
        
        # Actually, let's simplify for tuning demo: 
        # Similarity is key, but expensive to re-compute. 
        # For now, we assume interactions_df rows implies some "candidate generation" step happened before 
        # or we treat all interactions (pos/neg) as the candidate set for LTR training.
        
        # We need the similarity score.
        # Let's iterate users again.
        
        df['similarity_score'] = 0.0
        
        url_to_idx = {url: idx for idx, url in enumerate(tfidf_recommender.articles_df['url'])}
        
        print("Calculating similarity features...")
        # To make this fast:
        # 1. Get user profiles for ALL users
        # 2. Vectorize all items
        
        # It's better to process by user group.
        grouped = df.groupby('user_id')
        
        # We can't easily vectorize this loop without rewriting a lot. 
        # Let's use the loop from before but apply to DF.
        
        # Pre-compute profiles?
        # user_profiles = {}
        # for user_id in df['user_id'].unique():
        #     ...
        
        # Let's stick to the previous implementation logic but apply more robustly.
        for user_id, group_indices in grouped.groups.items():
            # Get user history urls from interactions_df (not just current df which might be split)
            # Actually we should use 'past' history. 
            # For simplicity: user profile is based on their POSITIVE clicks in this dataset.
            
            user_clicks = interactions_df[(interactions_df['user_id'] == user_id) & (interactions_df['clicked'] == 1)]
            user_history_urls = user_clicks['article_url']
            history_indices = [url_to_idx.get(u) for u in user_history_urls if u in url_to_idx]
            
            if not history_indices:
                continue
                
            user_profile = tfidf_recommender.get_user_profile(history_indices)
            
            # Now for every item in this group (pos and neg samples)
            # We compute sim
            group_urls = df.loc[group_indices, 'url']
            for idx_in_df, url in group_urls.items():
                if url in url_to_idx:
                    vec = tfidf_recommender.article_vectors[url_to_idx[url]]
                    sim = (user_profile @ vec.T).item()
                    df.at[idx_in_df, 'similarity_score'] = sim
                    
        df.sort_values(by='user_id', inplace=True)
        
        features = ['user_avg_read_time', 'user_total_clicks', 'item_popularity', 'item_avg_rating', 'similarity_score']
        self.feature_cols = features
        
        X = df[features]
        y = df['rating'] # Use rating (0-5) as label. 0 for negatives.
        groups = df.groupby('user_id').size().to_frame('size')['size'].to_numpy()
        
        return X, y, groups

    def tune_hyperparameters(self, X, y, groups):
        """
        Simple Grid Search for Hyperparameters.
        Due to Group structure, standard GridSearchCV is tricky with XGBRanker.
        We will do a simple manual search or use sk-learn GroupKFold if possible.
        """
        print("Tuning Hyperparameters...")
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100]
        }
        
        best_score = -1
        best_params = {}
        
        # Manual Grid Search with grouping
        import itertools
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Valid set split (Last 20% of users)
        # Groups array tells us size of each user's list.
        # We need to split X, y based on groups.
        
        n_groups = len(groups)
        split_idx = int(n_groups * 0.8)
        train_groups = groups[:split_idx]
        valid_groups = groups[split_idx:]
        
        train_size = sum(train_groups)
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        X_valid = X.iloc[train_size:]
        y_valid = y.iloc[train_size:]
        
        for params in combinations:
            model = xgb.XGBRanker(
                objective='rank:ndcg',
                min_child_weight=0.1,
                eval_metric='ndcg@5',
                **params
            )
            model.fit(
                X_train, y_train,
                group=train_groups,
                eval_set=[(X_valid, y_valid)],
                eval_group=[valid_groups],
                verbose=False
            )
            
            # Retrieve score from history
            results = model.evals_result()
            # results structure: {'validation_0': {'ndcg@5': [score1, score2, ...]}}
            score = results['validation_0']['ndcg@5'][-1]
            
            if score > best_score:
                best_score = score
                best_params = params
                
        print(f"Best Params: {best_params} | Best NDCG: {best_score}")
        self.model = xgb.XGBRanker(objective='rank:ndcg', min_child_weight=0.1, **best_params)

    def train(self, X, y, groups):
        print("Training XGBRanker...")
        self.model.fit(
            X, y,
            group=groups,
            verbose=True
        )
        print("Training complete.")

    def predict(self, X):
        return self.model.predict(X)
