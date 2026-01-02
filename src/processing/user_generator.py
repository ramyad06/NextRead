import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

class UserGenerator:
    def __init__(self, articles_path, output_dir="data/processed", n_users=100):
        self.articles_path = articles_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.articles_df = pd.read_csv(articles_path)
        self.articles_df['content'] = self.articles_df['content'].fillna('')
        self.articles_df['full_text'] = self.articles_df['title'] + " " + self.articles_df['content']
        self.n_users = n_users
        
    def generate(self):
        print("Vectorizing articles...")
        tfidf = TfidfVectorizer(stop_words='english', max_features=500)
        article_vectors = tfidf.fit_transform(self.articles_df['full_text'])
        
        users = []
        interactions = []
        
        print(f"Generating {self.n_users} users...")
        for user_id in range(self.n_users):
            # Assign a random interest vector (simulating a user profile)
            # We construct a user profile by averaging 3-5 random articles they "liked" initially
            n_interests = random.randint(3, 5)
            # Pick random indices
            interest_indices = np.random.choice(article_vectors.shape[0], n_interests, replace=False)
            
            # User vector is mean of these articles
            user_vector = np.mean(article_vectors[interest_indices], axis=0)
            user_vector = np.asarray(user_vector) # ensure array
            
            users.append({
                'user_id': user_id,
                'signup_date': '2025-01-01',
                # Store interest vector? No, usually we don't have it explicitly in prod, 
                # but for generation we rely on it.
            })
            
            # Simulate interactions for this user across ALL articles
            # Probability of click depends on similarity
            similarities = cosine_similarity(user_vector, article_vectors).flatten()
            
            # Interaction logic:
            # High similarity -> High prob of click
            for idx, score in enumerate(similarities):
                # Sigmoid-ish probability or threshold
                # Let's say prob = score (which is 0-1 range usually for cosine)
                # Add some noise
                prob = score
                if random.random() < prob:
                    # Clicked!
                    word_count = len(self.articles_df.iloc[idx]['full_text'].split())
                    max_seconds = (word_count / 200) * 60
                    read_time = random.uniform(0.5, 1.0) * max_seconds if score > 0.3 else random.uniform(0, 0.3) * max_seconds
                    
                    interactions.append({
                        'user_id': user_id,
                        'article_url': self.articles_df.iloc[idx]['url'],
                        'clicked': 1,
                        'read_time_seconds': int(read_time),
                        'rating': 5 if score > 0.5 else (4 if score > 0.3 else 3),
                        'timestamp': pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 30))
                    })
                else:
                    # Negative Sample (Implicit feedback: unseen or ignored)
                    # We sample negatives to keep dataset manageable (e.g., 20% prob of recording a negative)
                    if random.random() < 0.2:
                        interactions.append({
                            'user_id': user_id,
                            'article_url': self.articles_df.iloc[idx]['url'],
                            'clicked': 0,
                            'read_time_seconds': 0,
                            'rating': 0,
                            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 30))
                        })
                
        users_df = pd.DataFrame(users)
        interactions_df = pd.DataFrame(interactions)
        
        users_df.to_csv(os.path.join(self.output_dir, "users.csv"), index=False)
        interactions_df.to_csv(os.path.join(self.output_dir, "interactions.csv"), index=False)
        
        print(f"Generated {len(users_df)} users and {len(interactions_df)} interactions.")
        return interactions_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "data/raw/articles.csv"
        
    generator = UserGenerator(path)
    generator.generate()
