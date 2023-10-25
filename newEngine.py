    
#importing libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tmdbv3api import TMDb
from tmdbv3api import Movie

# Load the dataset 
ratings_file = '/Users/anuththaradivyanjalie/Documents/SLIIT/IRWA/Group Assignment/BeginnerBugs_MovieRecommendation/new_dataset/ratings.csv'  
movies_file = '/Users/anuththaradivyanjalie/Documents/SLIIT/IRWA/Group Assignment/BeginnerBugs_MovieRecommendation/new_dataset/movies.csv'    

df_ratings = pd.read_csv(ratings_file)
df_movies = pd.read_csv(movies_file)


# Additional Data Preprocessing
# Process movie titles to remove year information
#In the movies dataframe, movie title contains the year of the movie as well. 
# Since we don't need that information we will be remove that part from the movie titel.
df_movies['title'] = df_movies['title'].str.replace(r'\(\d{4}\)', '', regex=True)


# Combine genres into a single string
df_movies['genres'] = df_movies['genres'].str.replace('|', ' ')

# Merge the ratings and movies data
movie_data = pd.merge(df_ratings, df_movies, on='movieId')

#removing unnessary columns from the mergerd dataset
movie_data = movie_data.drop('timestamp', axis=1)

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
nltk.download('stopwords')

# Create a TF-IDF vectorizer for movie descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
df_movies['genres'] = df_movies['genres'].fillna('')  # Fill missing genres
tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['genres'])

# TMDb setup
tmdb = TMDb()
tmdb.api_key = '8265bd1679663a7ea12ac168da84d2e8&language=en-US'  # Replace with your TMDb API key

# Create a function to get movie recommendations and posters
def get_recommendations_with_posters(title, num_recommendations=5):
    idx = df_movies[df_movies['title'] == title].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    similar_movies = list(enumerate(cosine_similarities[0]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    similar_movies = similar_movies[1:num_recommendations + 1]
    recommended_movies = [df_movies['title'].iloc[i[0]] for i in similar_movies]

    recommended_movies_with_posters = []
    for movie_title in recommended_movies:
        poster_url = get_movie_poster(movie_title)
        recommended_movies_with_posters.append((movie_title, poster_url))
    
    return recommended_movies_with_posters

# Function to get movie poster from TMDb
def get_movie_poster(movie_title):
    movie = Movie()
    search = movie.search(movie_title)
    if search:
        movie_id = search[0]['id']  # Assuming the first search result is the correct movie
        movie_info = movie.details(movie_id)
        poster_path = movie_info['poster_path']
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:
        return None
    
# Create a Streamlit app
original_title = '<p style="font-family: Courier; color:LightPurple; font-size: 50px;">Movie Bugs</p>'
#Arial
st.markdown(original_title, unsafe_allow_html=True)
# st.title('Movie Bugs')
st.write('Welcome to Movie Bugs! This app provides movie recommendations based on genres and user ratings.')

# Input for user ID and movie name
user_id = st.number_input('Enter User ID', min_value=1)
movie_name = st.selectbox('Select a Movie', df_movies['title'])

# Show movie recommendations with posters if a movie name is provided
if movie_name:
    try:
        recommendations = get_recommendations_with_posters(movie_name)
        st.write(f'Recommended Movies for {movie_name}:')

        # Create five columns to display posters in rows of 5
        columns = st.columns(3)

        for i, (recommended_movie, poster_url) in enumerate(recommendations):
            col = columns[i % 3]  # Alternate between the 5 columns

            col.write(recommended_movie)
            if poster_url:
                col.image(poster_url, caption=recommended_movie, width=150)
    except KeyError:
        st.error('Movie not found in the dataset. Please try a different movie name.')



# Allow users to rate genre-based recommended movies and store ratings
if recommendations:  # Check if recommendations exist
    st.subheader('Rate Genre-Based Recommended Movies')
    ratings = {}
    for recommended_movie, poster_url in recommendations:
        rating = st.slider(f'Rate {recommended_movie}', min_value=0, max_value=5, value=0, key=recommended_movie)
        ratings[recommended_movie] = rating

    if st.button('Submit Genre-Based Ratings'):
        # Append new genre-based ratings to the ratings.csv file
        new_ratings = []
        for movie, rating in ratings.items():
            movie_id = df_movies[df_movies['title'] == movie]['movieId'].values[0]
            new_ratings.append({'userId': user_id, 'movieId': movie_id, 'rating': rating, 'timestamp': 0})

        df_new_ratings = pd.DataFrame(new_ratings)
        print(df_new_ratings)
        df_ratings = pd.concat([df_ratings, df_new_ratings], ignore_index=True)
        print(df_ratings)
        df_ratings.to_csv(ratings_file, sep=',', index=False, encoding='utf-8')

        st.success('Genre-based ratings submitted and stored in ratings.csv')





