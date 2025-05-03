import streamlit as st
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load preprocessed dataset
@st.cache_data
def load_data():
    df = pd.read_csv("movielens_cleaned.csv")
    return df

df = load_data()

# TF-IDF vectorization on genres
tfidf = TfidfVectorizer(stop_words='english')
df['genres'] = df['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Cosine similarity for content-based filtering
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# NearestNeighbors for user-based collaborative filtering
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
user_movie_matrix = pd.pivot_table(df, values='avg_rating', index='title', columns='movieId', fill_value=0)
nn_model.fit(user_movie_matrix.values)

# Content-based recommendations
def get_content_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df[['title', 'avg_rating']].iloc[movie_indices]

# Collaborative filtering recommendations
def get_collab_recommendations(title):
    idx = user_movie_matrix.index.get_loc(title)
    distances, indices_nn = nn_model.kneighbors([user_movie_matrix.iloc[idx]], n_neighbors=11)
    recommended_titles = user_movie_matrix.index[indices_nn[0][1:]].tolist()
    return df[df['title'].isin(recommended_titles)][['title', 'avg_rating']].drop_duplicates()

# Streamlit UI
st.title("🍿 Movie Recommendation System")

st.sidebar.header("🛠️ Filters")
min_rating = st.sidebar.slider("Minimum Average Rating", 0.0, 5.0, 3.0, step=0.1)
selected_genre = st.sidebar.selectbox("Select Genre (optional)", options=["All"] + sorted(df['genres'].str.split('|').explode().unique().tolist()))

filtered_df = df[df['avg_rating'] >= min_rating]
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['genres'].str.contains(selected_genre)]

st.subheader("🎥 Choose a Movie")
movie_list = filtered_df['title'].sort_values().unique().tolist()
selected_movie = st.selectbox("Select a movie:", movie_list)

# Tabs for model choice
tab1, tab2 = st.tabs(["🎯 Content-Based", "👥 User-Based"])

with tab1:
    if st.button("🎬 Recommend (Content-Based)"):
        recs = get_content_recommendations(selected_movie)
        st.subheader("🎯 Top 10 Genre-Based Recommendations")
        st.dataframe(recs.reset_index(drop=True))

with tab2:
    if st.button("👥 Recommend (User-Based)"):
        recs = get_collab_recommendations(selected_movie)
        st.subheader("👥 Top 10 Collaborative Recommendations")
        st.dataframe(recs.reset_index(drop=True))
