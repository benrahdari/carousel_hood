from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, session
from flask_session import Session
import pandas as pd
import numpy as np
import random
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
import json
import os
from flask_wtf import CSRFProtect
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "insight_general_ai_evaluation_system"

csrf = CSRFProtect(app)

# Configure session to use filesystem (instead of signed cookies)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Configuration for data paths
MOVIES_PATH = 'data/movie_metadata.json'  # Update with the correct path
RATINGS_PATH = 'data/ratings.csv'         # Update with the correct path

# MovieRecommender Class
class MovieRecommender:
    def __init__(self, movies_df, ratings_df, top_n=100):
        np.random.seed(42)
        random.seed(42)
        self.top_n = top_n
        self.movies_df = movies_df[movies_df['genres'] != '(no genres listed)'].copy()
        self._preprocess_data(ratings_df)
        self._build_interactions()
        self._train_model()

    def _preprocess_data(self, ratings_df):
        # Filter ratings for movies present in movies_df
        self.ratings_df = ratings_df[ratings_df['movieId'].isin(self.movies_df['movieId'])].copy()

        # Get top_n movies based on total ratings
        top_movies = (
            self.ratings_df.groupby('movieId')['rating']
            .sum()
            .sort_values(ascending=False)
            .head(self.top_n)
            .index.tolist()
        )
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(top_movies)].reset_index(drop=True)

        if self.ratings_df.empty:
            raise ValueError("No ratings available after preprocessing. Please check your ratings.csv and filtering criteria.")

        # Categorize users and items
        user_id_cats = pd.Categorical(self.ratings_df['userId'])
        item_id_cats = pd.Categorical(self.ratings_df['movieId'])
        self.ratings_df['user_id'] = user_id_cats.codes
        self.ratings_df['item_id'] = item_id_cats.codes
        self.user_id_map = dict(enumerate(user_id_cats.categories))
        self.item_id_map = dict(enumerate(item_id_cats.categories))
        self.reverse_user_id_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_item_id_map = {v: k for k, v in self.item_id_map.items()}

    def _build_interactions(self):
        self.interactions = Interactions(
            user_ids=self.ratings_df['user_id'].values,
            item_ids=self.ratings_df['item_id'].values,
            ratings=self.ratings_df['rating'].values.astype(np.float32)
        )
        self.train_interactions, self.test_interactions = random_train_test_split(
            self.interactions,
            test_percentage=0.2,
            random_state=np.random.RandomState(42)
        )

    def _train_model(self):
        self.model = ExplicitFactorizationModel(n_iter=10, random_state=np.random.RandomState(42))
        self.model.fit(self.train_interactions)
        self.train_rmse = rmse_score(self.model, self.train_interactions)
        self.test_rmse = rmse_score(self.model, self.test_interactions)
        print(f'Train RMSE: {self.train_rmse}')
        print(f'Test RMSE: {self.test_rmse}')

    def get_recommendations(self, user_id, recommendation_type='carousel', num_items_per_carousel=10, num_carousels=5):
        if user_id not in self.reverse_user_id_map:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        user_internal_id = self.reverse_user_id_map[user_id]
        n_items = len(self.item_id_map)
        item_internal_ids = np.arange(n_items)
        scores = self.model.predict(user_internal_id, item_internal_ids)

        # Fetch known positives from the dataset
        known_positives = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].unique()

        # Prepare recommendations DataFrame
        scores_df = pd.DataFrame({'movieId': list(self.item_id_map.values()), 'score': scores})
        scores_df = scores_df[~scores_df['movieId'].isin(known_positives)]
        recommendations = scores_df.merge(self.movies_df, on='movieId')

        if recommendation_type == 'carousel':
            return self._organize_recommendations_by_genre(recommendations, num_items_per_carousel, num_carousels)
        elif recommendation_type == 'list':
            return recommendations.sort_values(by='score', ascending=False).reset_index(drop=True)
        else:
            raise ValueError("Invalid recommendation_type. Choose 'carousel' or 'list'.")

    def _organize_recommendations_by_genre(self, recommendations, num_items_per_carousel=10, num_carousels=5):
        recommendations_genres = recommendations.copy()
        recommendations_genres['genres'] = recommendations_genres['genres'].str.split('|')
        recommendations_genres = recommendations_genres.explode('genres')
        genre_groups = recommendations_genres.groupby('genres')
        genre_recommendations = {}
        for genre, group in genre_groups:
            genre_recommendations[genre] = group.sort_values(by='score', ascending=False).head(num_items_per_carousel)
        sorted_genres = self._get_sorted_genres(genre_recommendations)
        sorted_genres = sorted_genres[:num_carousels]
        return {genre: genre_recommendations[genre].to_dict(orient='records') for genre in sorted_genres}

    def _get_sorted_genres(self, genre_recommendations):
        genre_scores = {genre: df['score'].sum() for genre, df in genre_recommendations.items()}
        return sorted(genre_scores, key=genre_scores.get, reverse=True)

def load_movies(json_path):
    with open(json_path, 'r') as f:
        movies = json.load(f)
    movies_df = pd.DataFrame(movies)
    expected_columns = {'movieId', 'Title', 'Year', 'imdbRating', 'Poster', 'genres'}
    
    missing = expected_columns - set(movies_df.columns)
    if missing:
        raise ValueError(f"Missing columns in movie_metadata.json: {missing}")
    
    return movies_df

def load_ratings(ratings_path):
    try:
        ratings_df = pd.read_csv(ratings_path)
        required_columns = {'userId', 'movieId', 'rating'}
        if not required_columns.issubset(ratings_df.columns):
            missing = required_columns - set(ratings_df.columns)
            raise ValueError(f"Missing columns in ratings.csv: {missing}")
        print("Ratings data loaded successfully.")
        return ratings_df
    except Exception as e:
        print(f"Error loading ratings data: {e}")
        raise

# Initialize recommender
movies_df = load_movies(MOVIES_PATH)
ratings_df = load_ratings(RATINGS_PATH)
recommender = MovieRecommender(movies_df, ratings_df)

# Helper function to update item scores based on user interaction
def update_item_scores(item_scores, clicked_item_id, seen_items, recommender, positive_factor=0.05, negative_factor=0.01, update_type='adaptive'):
    if update_type == 'control':
        # In control, do not update based on clicks
        return item_scores

    if update_type == 'adaptive':
        # Retrieve clicked item's genres
        clicked_item_row = recommender.movies_df[recommender.movies_df['movieId'] == clicked_item_id]
        if clicked_item_row.empty:
            print(f"Clicked item ID {clicked_item_id} not found in movies_df.")
            return item_scores  # Exit if movie not found

        clicked_item_genres = set(clicked_item_row['genres'].values[0].split('|'))

        # Collect all genres from seen items
        seen_genres = set()
        for item_id in seen_items:
            item_row = recommender.movies_df[recommender.movies_df['movieId'] == item_id]
            if not item_row.empty:
                seen_genres.update(item_row['genres'].values[0].split('|'))

        # Exclude genres present in the clicked item
        negative_genres = seen_genres - clicked_item_genres

        # Decrease scores for movies in negative genres
        for item_id, score in item_scores.items():
            item_row = recommender.movies_df[recommender.movies_df['movieId'] == item_id]
            if item_row.empty:
                continue  # Skip if movie not found

            item_genres = set(item_row['genres'].values[0].split('|'))

            # Check if movie has any negative genres
            if item_genres & negative_genres:
                item_scores[item_id] -= item_scores[item_id] * negative_factor
                if item_scores[item_id] < 0:
                    item_scores[item_id] = 0

            # Additionally, handle exact genre combination matches
            if item_genres == (clicked_item_genres - negative_genres):
                item_scores[item_id] -= item_scores[item_id] * negative_factor
                if item_scores[item_id] < 0:
                    item_scores[item_id] = 0

        # Increase scores for movies sharing genres with the clicked item
        for item_id, score in item_scores.items():
            if item_id == clicked_item_id:
                continue  # Skip the clicked item itself

            item_row = recommender.movies_df[recommender.movies_df['movieId'] == item_id]
            if item_row.empty:
                continue  # Skip if movie not found

            item_genres = set(item_row['genres'].values[0].split('|'))

            if clicked_item_genres & item_genres:
                item_scores[item_id] += item_scores[item_id] * positive_factor

        return item_scores

    raise ValueError("Invalid update_type. Choose 'adaptive' or 'control'.")

# Helper function to get baseline recommendations
def get_baseline_recommendations(recommender, baseline_type='popularity', num_items=50, recommendation_type='carousel', random_baseline=False):
    if baseline_type == 'random' or random_baseline:
        # Fetch random movies from the dataset
        recommendations = recommender.movies_df.sample(n=num_items, random_state=42).copy()
        recommendations['score'] = np.random.rand(num_items)
    elif baseline_type == 'popularity':
        # Calculate popularity based on total ratings
        popularity = (
            recommender.ratings_df.groupby('movieId')['rating']
            .sum()
            .sort_values(ascending=False)
            .head(num_items)
            .reset_index()
        )
        recommendations = recommender.movies_df[recommender.movies_df['movieId'].isin(popularity['movieId'])]
        recommendations = recommendations.merge(popularity, on='movieId')
        recommendations.rename(columns={'rating': 'score'}, inplace=True)
    else:
        raise ValueError("Invalid baseline_type. Choose 'random' or 'popularity'.")

    if recommendation_type == 'carousel':
        return recommender._organize_recommendations_by_genre(recommendations, num_items_per_carousel=10, num_carousels=5)
    elif recommendation_type == 'list':
        return recommendations.sort_values(by='score', ascending=False).reset_index(drop=True)
    else:
        raise ValueError("Invalid recommendation_type. Choose 'carousel' or 'list'.")

# Helper function to get current recommendations
def get_current_recommendations(item_scores, clicked_items, recommender, recommendation_type='list', num_items=50, random_update=False, update_type='adaptive'):
    if update_type == 'adaptive':
        # Adaptive logic already applied when updating item_scores
        pass
    elif update_type == 'control':
        # Do not modify item_scores
        pass
    else:
        raise ValueError("Invalid update_type. Choose 'adaptive' or 'control'.")

    # Create DataFrame from item_scores
    item_scores_df = pd.DataFrame({
        'movieId': list(item_scores.keys()),
        'score': list(item_scores.values())
    })

    # Merge with movies data
    recommendations = item_scores_df.merge(recommender.movies_df, on='movieId')

    # Exclude clicked items
    recommendations = recommendations[~recommendations['movieId'].isin(clicked_items)]

    if recommendation_type == 'list':
        recommendations = recommendations.sort_values(by='score', ascending=False).head(num_items)
        return recommendations
    elif recommendation_type == 'carousel':
        return recommender._organize_recommendations_by_genre(recommendations, num_items_per_carousel=10, num_carousels=5)
    else:
        raise ValueError("Invalid recommendation_type. Choose 'carousel' or 'list'.")


import random
import string


def generate_user_id():
    """Generates a user ID with the format: one digit - three letters - four digits."""
    return ''.join(random.choices(string.digits[1:], k=8))

# Landing page route
@app.route('/study', methods=['GET'])
@csrf.exempt
def landing_page():
    if 'parameters' not in session:
        # Get query parameters
        type_param = request.args.get('type', 'adaptive')          # 'adaptive' or 'control'
        baseline_param = request.args.get('baseline', 'popularity')  # 'random' or 'popularity'
        ui_param = request.args.get('ui', 'carousel')             # 'carousel' or 'list'
        mouse_logging_param = request.args.get('mouse_logging', 'true')
        mouse_logging_freq_param = request.args.get('mouse_logging_freq', '1000')  # Default 500ms

        # Set session parameters
        session['parameters'] = {
            'type': type_param,
            'baseline': baseline_param,
            'ui': ui_param,
            'mouse_logging': mouse_logging_param.lower() == 'true',
            'mouse_logging_freq': int(mouse_logging_freq_param),
        }

    params = session['parameters']
    recommendation_type = params['ui']          # 'carousel' or 'list'
    update_type = params['type']                # 'adaptive' or 'control'
    baseline_type = params['baseline']          # 'random' or 'popularity'


    if 'iteration' not in session:
        # First visit, initialize user state
        session['iteration'] = 1
        session['clicked_items'] = []
        # Generate a new user_id
        user_id = generate_user_id()
        session['user_id'] = user_id
        # Initialize item_scores based on baseline
        num_items = 500
        baseline_recommendations = get_baseline_recommendations(
            recommender,
            baseline_type=baseline_type,
            num_items=num_items,
            recommendation_type=recommendation_type,
            random_baseline=(baseline_type == 'random')
        )
        if recommendation_type == 'carousel':
            # Flatten the recommendations to create item_scores
            item_scores = {}
            for genre, items in baseline_recommendations.items():
                for item in items:
                    item_scores[int(item['movieId'])] = item['score']
        else:
            # For list, use the 'score' directly
            item_scores = {int(row['movieId']): row['score'] for _, row in baseline_recommendations.iterrows()}
        session['item_scores'] = item_scores
        user_id = session['user_id']
    else:
        # Subsequent visits, retain session state
        user_id = session['user_id']

    item_scores = session['item_scores']
    clicked_items = session['clicked_items']
    iteration = session['iteration']

    if iteration > 10:
        uid = session['user_id']
        return render_template('completion.html', uid=uid)

    # Generate current recommendations
    recommendations = get_current_recommendations(
        item_scores,
        clicked_items,
        recommender,
        recommendation_type=recommendation_type,
        num_items=50,
        random_update=False,
        update_type=update_type
    )

    # Store the seen items for use in the click handler
    if recommendation_type == 'carousel':
        seen_items = []
        for genre, items in recommendations.items():
            seen_items.extend([item['movieId'] for item in items])
        session['seen_items'] = seen_items
    else:
        session['seen_items'] = [item['movieId'] for _, item in recommendations.iterrows()]

    # Convert recommendations to appropriate format for rendering
    if recommendation_type == 'carousel':
        recommendations_dict = recommendations
    else:
        recommendations_dict = recommendations.to_dict(orient='records')

    # For debugging: get current session parameters
    current_parameters = session['parameters']

    # Render the recommendations
    return render_template(
        'recommendations.html',
        recommendations=recommendations_dict,
        iteration=iteration,
        recommendation_type=recommendation_type,
        current_parameters=current_parameters,
        user_id=user_id
    )

# Click handler route
@app.route('/click', methods=['POST'])
@csrf.exempt
def click():
    phase = request.form.get('phase', None)
    if phase == 'end_of_decision':
        # No movie selected, proceed to next iteration
        iteration = session.get('iteration', 1)
        session['iteration'] = iteration + 1
        # Check if the session should end
        if iteration >= 10:
            return redirect(url_for('completion'))
        else:
            return redirect(url_for('landing_page'))
    else:
        # Existing code for handling movie clicks
        clicked_item_id = int(request.form['movieId'])

        # Retrieve user state from session
        item_scores = session.get('item_scores', {})
        clicked_items = session.get('clicked_items', [])
        iteration = session.get('iteration', 1)
        user_id = session.get('user_id')
        seen_items = session.get('seen_items', [])
        params = session.get('parameters', {})
        recommendation_type = params.get('ui', 'carousel')
        update_type = params.get('type', 'adaptive')

        # Append clicked item to clicked_items
        clicked_items.append(clicked_item_id)
        session['clicked_items'] = clicked_items

        # Remove clicked_item_id from seen_items
        if clicked_item_id in seen_items:
            seen_items.remove(clicked_item_id)
        session['seen_items'] = seen_items

        # Update item_scores based on update_type
        item_scores = update_item_scores(
            item_scores,
            clicked_item_id,
            seen_items,
            recommender,
            positive_factor=0.05,
            negative_factor=0.01,
            update_type=update_type
        )
        session['item_scores'] = item_scores

        # Increment iteration
        session['iteration'] = iteration + 1

        # Redirect back to landing page with current parameters
        if session['iteration'] >= 10:
            return redirect(url_for('completion'))
        else:
            return redirect(url_for('landing_page'))

# Completion page route
@app.route('/completion', methods=['GET'])
@csrf.exempt
def completion():
    uid = session['user_id']
    return render_template('completion.html',uid=uid)

# Route to clear the session
@app.route('/clear_session', methods=['GET'])
@csrf.exempt
def clear_session_route():
    session.clear()  # Removes all session data
    flash('Your session has been completely reset.')
    return redirect(url_for('landing_page'))

# Route to handle data submission
@app.route('/submit_data', methods=['POST'])
@csrf.exempt
def submit_data():
    data = request.get_json(force=True)

    if data is None:
        print("No data received")
        return jsonify({"message": "No data received"}), 400

    user_id = data.get('user_id')
    folder = 'experiment_data/'  
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    # Save the data to a JSON file
    filename = os.path.join(folder, f'{user_id}.json')
    with open(filename, 'w') as file:
        json.dump(data, file)
    # Return a response
    return jsonify({"message": "Data received and saved successfully"}), 200


# Route to handle survey data submission
@app.route('/submit_survey', methods=['POST'])
@csrf.exempt
def submit_survey():
    data = request.get_json(force=True)

    if data is None:
        print("No data received")
        return jsonify({"message": "No data received"}), 400

    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"message": "User ID is missing"}), 400

    # Prepare the folder for saving data
    folder = 'experiment_data/survey'  
    os.makedirs(folder, exist_ok=True)

    # Save the data to a JSON file named after the user_id
    filename = os.path.join(folder, f'{user_id}_survey.json')
    with open(filename, 'w') as file:
        json.dump(data, file)

    # Return a success response
    return jsonify({"message": "Survey data received and saved successfully"}), 200

@app.route('/show_params', methods=['GET'])
def show_params():
    # Pass Flask context variables explicitly
    context = {
        'session': dict(session),  # Convert session to a regular dictionary
        'args': request.args.to_dict(),  # Query parameters (GET)
        'form': request.form.to_dict(),  # Form data (POST)
        'json_body': request.get_json(silent=True)  # JSON body (POST)
    }
    return render_template('show_params.html', context=context)


GROUP_FILE = 'experiment_data/groups/group_data.json'

# Initialize the groups
def load_groups():
    if os.path.exists(GROUP_FILE):
        with open(GROUP_FILE, 'r') as f:
            return json.load(f)
    else:
        return [0] * 8  # 8 groups for the 2x2x2 factorial design

def save_groups(groups):
    with open(GROUP_FILE, 'w') as f:
        json.dump(groups, f)

def balanced_assign_participant(groups):
    min_index = np.argmin(groups)
    groups[min_index] += 1
    save_groups(groups)  # Save updated groups after assignment
    return min_index

@app.route('/start')
@csrf.exempt  # Exempt this route from CSRF protection
def assign_participant():
    groups = load_groups()  # Load the current group distribution
    group_index = balanced_assign_participant(groups)
    
    # Mapping based on group index
    mapping = {
        0: ('control', 'random', 'list'),
        1: ('control', 'random', 'carousel'),
        2: ('control', 'popularity', 'list'),
        3: ('control', 'popularity', 'carousel'),
        4: ('adaptive', 'random', 'list'),
        5: ('adaptive', 'random', 'carousel'),
        6: ('adaptive', 'popularity', 'list'),
        7: ('adaptive', 'popularity', 'carousel')
    }
    
    # Get the mapped parameters based on the group index
    type_param, baseline_param, ui_param = mapping[group_index]

    # Construct the redirect URL
    redirect_url = url_for('landing_page', type=type_param, baseline=baseline_param, ui=ui_param)
    
    # Redirect to the /study route with the selected parameters
    return redirect(redirect_url)