import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
@st.cache_data
def load_data():
    ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", 
                         names=["movieId", "title", "release_date", "video_release_date", "IMDb_URL",
                                "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"],
                         usecols=["movieId", "title"])
    return ratings, movies

ratings, movies = load_data()

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

# Compute similarities
@st.cache_data
def compute_similarities():
    user_similarity = pd.DataFrame(cosine_similarity(user_item_matrix),
                                   index=user_item_matrix.index, columns=user_item_matrix.index)
    item_similarity = pd.DataFrame(cosine_similarity(user_item_matrix.T),
                                   index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return user_similarity, item_similarity

user_similarity, item_similarity = compute_similarities()

# Recommendation Functions

def user_based_recommendation(user_id, top_n=5):
    similar_user = user_similarity[user_id].sort_values(ascending=False)[1:6].index
    similar_user_ratings = user_item_matrix.loc[similar_user].mean(axis=0)
    user_rate_movies = user_item_matrix.loc[user_id]
    unseen_movies = similar_user_ratings[user_rate_movies == 0]
    recs = unseen_movies.sort_values(ascending=False).head(top_n)
    return movies[movies["movieId"].isin(recs.index)][["movieId", "title"]]

def item_based_recommendation(user_id, top_n=5):
    user_ratings = user_item_matrix.loc[user_id].copy()
    pred_ratings = pd.Series(0.0, index=user_item_matrix.columns)
    for movie_id in user_item_matrix.columns:
        if user_ratings[movie_id] == 0:
            sim_scores = item_similarity[movie_id]
            rated_mask = user_ratings > 0
            if sim_scores[rated_mask].sum() > 0:
                pred_ratings[movie_id] = np.dot(sim_scores[rated_mask], user_ratings[rated_mask]) / sim_scores[rated_mask].sum()
    top_movies = pred_ratings.sort_values(ascending=False).head(top_n)
    return movies[movies["movieId"].isin(top_movies.index)][["movieId", "title"]]

def svd_recommendation(user_id, top_n=5):
    R = user_item_matrix.values
    U, sigma, Vt = np.linalg.svd(R, full_matrices=False)
    sigma_matrix = np.diag(sigma[:50])
    R_hat = np.dot(np.dot(U[:, :50], sigma_matrix), Vt[:50, :])
    pred_ratings_matrix = pd.DataFrame(R_hat, index=user_item_matrix.index, columns=user_item_matrix.columns)
    user_row = pred_ratings_matrix.loc[user_id].copy()
    already_rated = user_item_matrix.loc[user_id] > 0
    user_row[already_rated] = 0
    top_movies = user_row.sort_values(ascending=False).head(top_n)
    return movies[movies["movieId"].isin(top_movies.index)][["movieId", "title"]]


# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.markdown("Choose a user and recommendation method to get personalized movie suggestions.")

user_id = st.number_input("Enter User ID (1 - 943):", min_value=1, max_value=943, value=1)
method = st.selectbox("Select Recommendation Method", ["User-User CF", "Item-Item CF", "SVD CF"])
top_n = st.slider("Number of Recommendations", 3, 15, 5)

if st.button("Get Recommendations"):
    if method == "User-User CF":
        recs = user_based_recommendation(user_id, top_n)
    elif method == "Item-Item CF":
        recs = item_based_recommendation(user_id, top_n)
    else:
        recs = svd_recommendation(user_id, top_n)
    
    st.subheader(f"Top {top_n} Recommended Movies for User {user_id}")
    st.dataframe(recs.reset_index(drop=True))

    if recs.empty:
        st.warning("No recommendations found. Try a different ID or method.")

