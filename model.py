import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
movies = pd.read_csv('data/tmdb_5000_movies.csv')
movies = movies[['id', 'title', 'overview']]
movies['overview'] = movies['overview'].fillna('')
movies = movies.drop_duplicates(subset='title')
movies = movies.reset_index(drop=True)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
print(f"Matrix shape: {tfidf_matrix.shape}")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movies.to_pickle('movies.pkl')
with open('cosine_sim.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

print("Model saved successfully!")
print(f"Total movies: {len(movies)}")