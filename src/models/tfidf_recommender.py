import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRecommender:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        self.article_vectors = None
        self.articles_df = None
        
    def fit(self, articles_df):
        """
        Fits the TF-IDF model on the articles dataframe.
        articles_df must have a 'full_text' column or 'title' + 'content'.
        """
        self.articles_df = articles_df.reset_index(drop=True)
        # Ensure full_text exists
        if 'full_text' not in self.articles_df.columns:
            self.articles_df['full_text'] = (self.articles_df['title'].fillna('') + " " + 
                                           self.articles_df['content'].fillna(''))
            
        print("Fitting TF-IDF vectorizer...")
        self.article_vectors = self.vectorizer.fit_transform(self.articles_df['full_text'])
        print(f"Fitted on {len(self.articles_df)} articles.")
        
    def get_user_profile(self, interaction_history_indices):
        """
        Computes a user profile vector based on the indices of articles they have interacted with.
        interaction_history_indices: list of indices in self.articles_df
        """
        if not interaction_history_indices:
            return np.zeros((1, self.article_vectors.shape[1]))
            
        # Get vectors for read articles
        read_vectors = self.article_vectors[interaction_history_indices]
        # Average them to get user profile
        user_vector = np.mean(read_vectors, axis=0)
        return np.asarray(user_vector)
        
    def get_candidates(self, user_profile_vector, top_k=50, exclude_indices=None):
        """
        Retrieves top_k similar articles given a user profile vector.
        """
        if user_profile_vector.ndim == 1:
            user_profile_vector = user_profile_vector.reshape(1, -1)
            
        # Compute cosine similarity
        similarities = cosine_similarity(user_profile_vector, self.article_vectors).flatten()
        
        # Sort indices
        sorted_indices = np.argsort(similarities)[::-1]
        
        candidates = []
        count = 0
        for idx in sorted_indices:
            if exclude_indices and idx in exclude_indices:
                continue
                
            candidates.append({
                'article_index': idx,
                'article_url': self.articles_df.iloc[idx]['url'],
                'similarity_score': similarities[idx]
            })
            count += 1
            if count >= top_k:
                break
                
        return candidates
