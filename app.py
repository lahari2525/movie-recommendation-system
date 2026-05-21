import streamlit as st
import pandas as pd
import pickle
import requests
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)
@st.cache_data
def load_data():
    movies = pd.read_pickle('movies.pkl')
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    return movies, cosine_sim

movies, cosine_sim = load_data()


def get_recommendations(movie_title, n=5):
    matches = movies[movies['title'] == movie_title]
    if matches.empty:
        return []
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

st.title("🎬 Movie Recommendation System")
st.markdown("Pick a movie you love, get 5 similar recommendations instantly.")

selected_movie = st.selectbox(
    "Search or select a movie:",
    sorted(movies['title'].tolist())
)
if st.button("🔍 Get Recommendations", type="primary"):
    recommendations = get_recommendations(selected_movie)
    
    if not recommendations:
        st.error("Movie not found. Try another title.")
    else:
        st.subheader(f"Movies similar to **{selected_movie}**:")
        
        cols = st.columns(5)
        
        for i, movie_title in enumerate(recommendations):
            with cols[i]:
                st.markdown(f"### 🎬")
                st.caption(movie_title)

st.markdown("---")
st.markdown("Built with Scikit-learn + Streamlit | Content-based filtering using TF-IDF")