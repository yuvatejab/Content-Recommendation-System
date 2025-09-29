import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, users_path, posts_path, engagements_path):
        self.users_df = pd.read_csv(users_path)
        self.posts_df = pd.read_csv(posts_path)
        self.engagements_df = pd.read_csv(engagements_path)
        self.interest_match_scores = self._calculate_scores()

    def _calculate_scores(self):
       
        tfidf_vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.8)
        user_interests_matrix = tfidf_vectorizer.fit_transform(self.users_df['top_3_interests'].str.replace(',', ''))

      
        post_tags_matrix = tfidf_vectorizer.transform(self.posts_df['tags'].str.replace(',', ''))

        
        return cosine_similarity(user_interests_matrix, post_tags_matrix)

    def get_recommendations(self, user_id, alpha=0.7, beta=0.3, top_n=3):
        try:
            user_index = self.users_df[self.users_df['user_id'] == user_id].index[0]
            user_engagement_score = self.users_df.loc[user_index, 'past_engagement_score']
        except IndexError:
            return [], pd.DataFrame() 

        user_interest_scores = self.interest_match_scores[user_index]
        hybrid_scores = (alpha * user_interest_scores) + (beta * user_engagement_score)

        recommendations_df = pd.DataFrame({
            'post_id': self.posts_df['post_id'],
            'hybrid_score': hybrid_scores
        })

        engaged_posts = self.engagements_df[self.engagements_df['user_id'] == user_id]['post_id']
        recommendations_df = recommendations_df[~recommendations_df['post_id'].isin(engaged_posts)]

        top_recommendations = recommendations_df.sort_values(by='hybrid_score', ascending=False).head(top_n)
        
       
        recommended_posts_details = self.posts_df[self.posts_df['post_id'].isin(top_recommendations['post_id'])]

        return top_recommendations['post_id'].tolist(), recommended_posts_details