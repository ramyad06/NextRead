import pandas as pd

class PopularityRecommender:
    def __init__(self):
        self.popular_articles = None
        
    def fit(self, interactions_df, articles_df):
        """
        Fits the model by calculating global popularity (click count).
        """
        # Count clicks per article
        pop_counts = interactions_df[interactions_df['clicked'] == 1].groupby('article_url').size().reset_index(name='click_count')
        
        # Merge with article info
        self.popular_articles = pop_counts.merge(articles_df, left_on='article_url', right_on='url', how='left')
        
        # Sort by popularity
        self.popular_articles = self.popular_articles.sort_values(by='click_count', ascending=False)
        
    def recommend(self, top_k=5):
        """
        Returns top K popular articles.
        """
        if self.popular_articles is None or self.popular_articles.empty:
            return []
            
        recs = []
        for i, row in self.popular_articles.head(top_k).iterrows():
            recs.append({
                'url': row['article_url'],
                'title': row['title'],
                'score': row['click_count']
            })
        return recs
