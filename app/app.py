import streamlit as st
import pandas as pd
from recommender import Recommender
from pathlib import Path


try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
   
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
USERS_PATH = DATA_DIR / "Users.csv"
POSTS_PATH = DATA_DIR / "Posts.csv"
ENGAGEMENTS_PATH = DATA_DIR / "Engagements.csv"



st.set_page_config(
    page_title="Content Recommendation System",
    page_icon="✨",
    layout="wide"
)


@st.cache_resource
def load_recommender():
   
    recommender = Recommender(
        users_path=USERS_PATH,
        posts_path=POSTS_PATH,
        engagements_path=ENGAGEMENTS_PATH
    )
    return recommender

recommender = load_recommender()
users_df = recommender.users_df


st.title(" Personalized Content Recommendation System")
st.markdown("Select a user to see their top 3 recommended posts based on their interests and engagement.")

col1, col2 = st.columns([1, 2])

with col1:
   
    user_list = users_df['user_id'].unique()
    selected_user = st.selectbox("Select a User ID", user_list)

   
    if selected_user:
        st.write("---")
        st.subheader(f"👤 User Profile: {selected_user}")
        user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
        st.markdown(f"**Interests:** `{user_info['top_3_interests']}`")
        st.markdown(f"**Engagement Score:** `{user_info['past_engagement_score']}`")

with col2:
  
    if selected_user:
        st.subheader("🌟 Top 3 Recommendations")
        
        recommended_post_ids, recommended_posts_details = recommender.get_recommendations(selected_user)

        if not recommended_post_ids:
            st.warning("No recommendations found for this user.")
        else:
            for index, row in recommended_posts_details.iterrows():
                with st.container(border=True):
                    st.markdown(f"#### Post ID: {row['post_id']}")
                    st.markdown(f"**Content Type:** {row['content_type'].capitalize()}")
                    st.markdown(f"**Tags:** `{row['tags']}`")


st.write("---")
with st.expander("🤔 How does this work?"):
    st.markdown("""
    This recommendation system uses a **hybrid model**:
    1.  **Content-Based Filtering:** It matches the user's `top_3_interests` with the `tags` on each post using a sophisticated text analysis model (TF-IDF and Cosine Similarity).
    2.  **User Engagement Heuristic:** It boosts recommendations for users with a higher `past_engagement_score`, personalizing the results.
    The final recommendation is a weighted combination of these two factors.
    """)