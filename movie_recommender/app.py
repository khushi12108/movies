from flask import Flask, render_template, request
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
API_KEY = os.getenv('API_KEY')
BASE_URL = 'https://api.themoviedb.org/3'

# Genre IDs
genre_ids = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35, "crime": 80,
    "documentary": 99, "drama": 18, "family": 10751, "fantasy": 14, "history": 36,
    "horror": 27, "music": 10402, "mystery": 9648, "romance": 10749,
    "science fiction": 878, "thriller": 53, "war": 10752, "western": 37,
}

# Load movie data
def load_movie_data():
    all_movies = []
    try:
        for page in range(1, 21):  # Fetch movies from pages 1 to 20 (larger dataset)
            url = f"{BASE_URL}/discover/movie"
            params = {
                'api_key': API_KEY,
                'sort_by': 'popularity.desc',
                'vote_average.gte': 7,  # Only include movies with a rating of 7 or higher
                'primary_release_date.gte': '1970-01-01',  # Start year
                'primary_release_date.lte': '2024-12-31',  # End year
                'page': page,
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            all_movies.extend(data.get('results', []))
        
        df = pd.DataFrame(all_movies)
        print(f"Data loaded: {df.shape[0]} movies")

        # Handle invalid release dates
        df['release_date'] = df['release_date'].fillna('1900-01-01')  # Fill missing dates with a default value
        df['release_year'] = pd.to_numeric(df['release_date'].str[:4], errors='coerce')

        # Exclude future dates (if any)
        current_year = pd.Timestamp.now().year
        df = df[df['release_year'] <= current_year]
        print("Sample release dates after excluding future dates:", df['release_date'].head())  # Debug: Print sample release dates
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie data: {e}")
        return pd.DataFrame()

# Fuzzy match movie title
def fuzzy_match_movie_title(movie_title, df):
    titles = df['title'].str.lower().tolist()
    print(f"Matching title: {movie_title}")  # Debugging print
    match, score = process.extractOne(movie_title.lower(), titles)
    print(f"Match found: {match} with score {score}")  # Debugging print
    return match if score >= 40 else None  # Lowered threshold to 40

# Content-based filtering using movie overviews
def get_content_based_recommendations(movie_title, df, top_n=10):
    # Ensure df is a DataFrame and not empty
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Invalid or empty DataFrame.")
        return pd.DataFrame()

    # Check for required columns
    required_columns = ['title', 'overview', 'vote_average', 'poster_path', 'release_date']
    if not all(col in df.columns for col in required_columns):
        print("Error: Missing required columns in DataFrame.")
        return pd.DataFrame()

    # Fuzzy match the movie title
    matched_title = fuzzy_match_movie_title(movie_title, df)
    if not matched_title:
        print("No match found for the movie title.")
        return pd.DataFrame()

    # Find matching movies
    matching_movies = df[df['title'].str.lower() == matched_title.lower()]
    if matching_movies.empty:
        print("No matching movies found.")
        return pd.DataFrame()

    # Get the index of the first matching movie
    idx = matching_movies.index[0]

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    df['overview'] = df['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Ensure idx is within bounds
    if idx >= len(cosine_sim):
        print(f"Error: Index {idx} is out of bounds for cosine_sim with size {len(cosine_sim)}")
        return pd.DataFrame()

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]  # Exclude the movie itself

    # Get top N similar movies
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices]

    # Format recommendations
    recommendations = recommendations.assign(
        poster=lambda x: x['poster_path'].apply(lambda y: f"https://image.tmdb.org/t/p/w500{y}" if y else ''),
        rating=lambda x: x['vote_average']
    )
    return recommendations[['title', 'poster', 'overview', 'rating', 'release_date']]

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# User Input Page
@app.route('/input', methods=['GET', 'POST'])
def input_movies():
    if request.method == 'POST':
        genre = request.form.get('genre')
        watched_movies = [movie.strip() for movie in request.form.get('movies', '').split(',') if movie.strip()]
        release_year = request.form.get('release_year', '')

        if not watched_movies:
            return render_template('recommendations.html', recommendations=[], message="Please enter at least one movie.")

        # Load movie data
        df = load_movie_data()
        print(f"Data loaded: {df.shape[0]} movies")

        # Check if the DataFrame is valid
        if not isinstance(df, pd.DataFrame) or df.empty:
            return render_template('recommendations.html', recommendations=[], message="Error fetching movie data.")

        # Check for required columns
        required_columns = ['title', 'overview', 'vote_average', 'poster_path', 'release_date', 'genre_ids']
        if not all(col in df.columns for col in required_columns):
            return render_template('recommendations.html', recommendations=[], message="Data error.")

        # Filter by genre (optional)
        if genre:
            genre_id = genre_ids.get(genre.lower())
            if genre_id:
                # Check if the genre_id is in the list of genre_ids for the movie
                df = df[df['genre_ids'].apply(lambda x: isinstance(x, list) and genre_id in x)]
                print(f"Filtered by genre: {genre} -> {df.shape[0]} movies")
            else:
                print(f"Invalid genre: {genre}")
        else:
            print("No genre filter applied.")

        # Filter by release year (optional)
        if release_year and '-' in release_year:
            start_year, end_year = map(int, release_year.split('-'))
            df['release_year'] = pd.to_numeric(df['release_date'].str[:4], errors='coerce')
            print("Release years before filtering:", df['release_year'].unique())  # Debug: Print unique release years
            df = df[(df['release_year'] >= start_year) & (df['release_year'] <= end_year)]
            print(f"Filtered by year: {release_year} -> {df.shape[0]} movies")
            print("Release years after filtering:", df['release_year'].unique())  # Debug: Print unique release years

            # Check if the DataFrame is empty after filtering
            if df.empty:
                return render_template('recommendations.html', recommendations=[], message="No movies found for the selected release year range.")
        else:
            print("No release year filter applied.")

        # Reset the DataFrame indices after filtering
        df = df.reset_index(drop=True)

        # Generate recommendations
        recommendations = pd.DataFrame()
        for movie_title in watched_movies:
            print(f"Finding recommendations for: {movie_title}")
            movie_recommendations = get_content_based_recommendations(movie_title, df)
            if not movie_recommendations.empty:
                print(f"Recommendations found for {movie_title}: {movie_recommendations.shape[0]} movies")
                recommendations = pd.concat([recommendations, movie_recommendations])
            else:
                print(f"No recommendations found for {movie_title}.")

        # Remove duplicates and sort by rating
        recommendations = recommendations.drop_duplicates(subset=['title'])
        if not recommendations.empty and 'rating' in recommendations.columns:
            recommendations = recommendations.sort_values(by='rating', ascending=False)

        # Render the recommendations page
        if recommendations.empty:
            return render_template('recommendations.html', recommendations=[], message="No recommendations found. Please try different filters.")
        return render_template('recommendations.html', recommendations=recommendations.to_dict('records'))

    return render_template('input.html')

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=False)  # Set debug=False for production