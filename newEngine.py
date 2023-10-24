import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load the dataset and perform any data preprocessing if necessary
ratings_file = 'ratings.csv'  # Update with your file path
movies_file = 'movies.csv'    # Update with your file path

df_ratings = pd.read_csv(ratings_file)
df_movies = pd.read_csv(movies_file)

# Merge the ratings and movies data
movie_data = pd.merge(df_ratings, df_movies, on='movieId')

# Create a user-movie rating pivot table
user_movie_rating = movie_data.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with zeros (unrated movies)
user_movie_rating = user_movie_rating.fillna(0)

# Standardize user ratings to have zero mean and unit variance
scaler = StandardScaler()
user_movie_rating_scaled = scaler.fit_transform(user_movie_rating)

# Calculate the similarity matrix using cosine similarity
cosine_sim = cosine_similarity(user_movie_rating_scaled)

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Create a function to get movie recommendations
def get_recommendations(title, num_recommendations=10):
    # Create a DataFrame for movie titles and their corresponding indices
    movie_indices = pd.Series(df_movies.index, index=df_movies['title'])
    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = df_movies['title'].iloc[movie_indices[1:num_recommendations + 1]]
    recommended_posters = [fetch_poster(df_movies['movieId'].iloc[idx]) for idx in movie_indices[1:num_recommendations + 1]]
    return recommended_movies, recommended_posters

# Create a Streamlit app
st.title('Movie Recommendation App')

# Input for user ID and movie name
user_id = st.number_input('Enter User ID', min_value=1)
movie_name = st.selectbox('Select a Movie', df_movies['title'])

# Show movie recommendations if a movie name is provided
if movie_name:
    try:
        recommendations = get_recommendations(movie_name)
        st.write(f'Recommended Movies for {movie_name}:')
        st.write(recommendations)
    except KeyError:
        st.error('Movie not found in the dataset. Please try a different movie name.')

# Use the Nearest Neighbors algorithm for user-specific recommendations
if movie_name and user_id:
    movie_id = df_movies[df_movies['title'] == movie_name]['movieId'].values[0]
    user_ratings = user_movie_rating_scaled[user_id - 1].reshape(1, -1)
    
    # Fit a Nearest Neighbors model using the user-movie rating matrix
    model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    model.fit(user_movie_rating_scaled)
    
    distances, indices = model.kneighbors(user_ratings, 10)
    recommended_movie_indices = indices[0]
    recommended_movies = [df_movies['title'].iloc[idx] for idx in recommended_movie_indices]
    st.write(f'User-Specific Recommendations for User {user_id}:')
    st.write(recommended_movies)







