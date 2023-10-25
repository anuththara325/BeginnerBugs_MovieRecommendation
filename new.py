import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset and perform any data preprocessing if necessary
ratings_file = 'ratings.csv'  # Update with your file path
movies_file = 'movies.csv'    # Update with your file path

df_ratings = pd.read_csv(ratings_file)
df_movies = pd.read_csv(movies_file)

# Additional Data Preprocessing
# Process movie titles to remove year information
df_movies['title'] = df_movies['title'].str.replace('(\(\d\d\d\d\))', '')
# Combine genres into a single string
df_movies['genres'] = df_movies['genres'].str.replace('|', ' ')

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

# Import NLTK and download stopwords
import nltk
nltk.download('stopwords')

# Create a TF-IDF vectorizer for movie descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
df_movies['genres'] = df_movies['genres'].fillna('')  # Fill missing genres
tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['genres'])

# Create a function to get movie recommendations
def get_recommendations(title, num_recommendations=10):
    # Create a DataFrame for movie titles and their corresponding indices
    movie_indices = pd.Series(df_movies.index, index=df_movies['title'])
    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores]
    return df_movies['title'].iloc[movie_indices[1:num_recommendations + 1]]

# Create a function to get movie recommendations based on genres
def get_genre_recommendations(title, num_recommendations=10):
    idx = df_movies[df_movies['title'] == title].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    similar_movies = list(enumerate(cosine_similarities[0]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    similar_movies = similar_movies[1:num_recommendations + 1]
    recommended_movies = [df_movies['title'].iloc[i[0]] for i in similar_movies]
    return recommended_movies

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

# Show genre-based movie recommendations
if movie_name:
    st.subheader('Genre-Based Recommendations')
    genre_recommendations = get_genre_recommendations(movie_name)
    st.write(f'Recommended Movies based on Genres for {movie_name}:')
    st.write(genre_recommendations)

# Use the Nearest Neighbors algorithm for user-specific recommendations
if movie_name and user_id:
    movie_id = df_movies[df_movies['title'] == movie_name]['movieId'].values[0]
    model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    model.fit(user_movie_rating_scaled)
    user_ratings = user_movie_rating_scaled[user_id - 1].reshape(1, -1)
    distances, indices = model.kneighbors(user_ratings, 10)
    recommended_movie_indices = indices[0]
    recommended_movies = [df_movies['title'].iloc[idx] for idx in recommended_movie_indices]
    st.write(f'User-Specific Recommendations for User {user_id}:')
    st.write(recommended_movies)

# Allow users to rate genre-based recommended movies and store ratings
if genre_recommendations:
    st.subheader('Rate Genre-Based Recommended Movies')
    ratings = {}
    for movie in genre_recommendations:
        rating = st.slider(f'Rate {movie}', min_value=0, max_value=5, value=0, key=movie)
        ratings[movie] = rating

    if st.button('Submit Genre-Based Ratings'):
        # Append new ratings to the ratings.csv file
        new_ratings = []
        for movie, rating in ratings.items():
            movie_id = df_movies[df_movies['title'] == movie]['movieId'].values[0]
            new_ratings.append({'userId': user_id, 'movieId': movie_id, 'rating': rating,'timestamp':0})
        
        df_new_ratings = pd.DataFrame(new_ratings)
        df_ratings = pd.concat([df_ratings, df_new_ratings], ignore_index=True)
        df_ratings.to_csv(ratings_file, index=False)
        st.success('Genre-based ratings submitted and stored in ratings.csv')