import pickle
import streamlit as st
import requests
import pandas as pd

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters

st.header('Movie Recommender System Using Machine Learning')
movies = pickle.load(open('artifacts/movie_list.pkl','rb'))
similarity = pickle.load(open('artifacts/similarity.pkl','rb'))

# Initialize a DataFrame to store user data
user_data = pd.DataFrame(columns=['Username', 'Movie Searched', 'Recommended Movie', 'Rating'])

# Allow the user to input their username and rate movies
user_name = st.text_input("Enter Your Username")
user_ratings = {}      

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie to get recommendation",
    movie_list
)

# if st.button('Show Recommendation'):
#     recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.text(recommended_movie_names[0])
#         st.image(recommended_movie_posters[0])
#     with col2:
#         st.text(recommended_movie_names[1])
#         st.image(recommended_movie_posters[1])

#     with col3:
#         st.text(recommended_movie_names[2])
#         st.image(recommended_movie_posters[2])
#     with col4:
#         st.text(recommended_movie_names[3])
#         st.image(recommended_movie_posters[3])
#     with col5:
#         st.text(recommended_movie_names[4])
#         st.image(recommended_movie_posters[4])

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)


    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
        
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
        

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
        
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
        
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
        
    # Add a section for user ratings
    user_ratings_title = "Rate Recommended Movies"
    st.markdown(f"### {user_ratings_title}")
    
      # Allow the user to rate the recommended movies

    for i in range(5):

        rating = st.slider(f"Rate {recommended_movie_names[i]}", 1, 5)

        user_data = pd.concat([user_data, pd.DataFrame({'Username': [user_name], 'Movie Searched': [selected_movie], 'Recommended Movie': [recommended_movie_names[i]], 'Rating': [rating]})], ignore_index=True)


if st.button('Show My Picks'):
    # Filter movies with ratings greater than 3
    high_rated_movies = user_data[user_data['Rating'] > 3]

    if not high_rated_movies.empty:
        st.markdown("### Your Highly Rated Movies:")
        st.dataframe(high_rated_movies)
    else:
        st.markdown("You haven't rated any movies with a rating greater than 3 yet.")


# Save user data to a CSV file
if st.button('Save Data to CSV'):
    user_data.to_csv('user_data.csv', index=False)

# Display the user data
st.dataframe(user_data)



