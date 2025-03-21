import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup paths
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load Reddit comment data
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    comments_df = pd.DataFrame(data['comments'])

# Prepare TF-IDF
texts = comments_df['text'].fillna("").tolist()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Cosine similarity search
def search_comments_cosine(query, top_n=20):
    if not query.strip():
        return []

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        row = comments_df.iloc[idx]
        score = round(float(similarities[idx]), 4)
        results.append({
            "author": row["author"],
            "text": row["text"],
            "depth": int(row["depth"]),
            "score": score  # ← TF-IDF cosine similarity score
        })
    return results

# Routes
@app.route("/")
def home():
    return render_template('base.html', title="Reddit Movie Comment Search")

@app.route("/search")
def search_route():
    query = request.args.get("q", "")
    return jsonify(search_comments_cosine(query))

# Run server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)






























'''
import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd

# Setup
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load Reddit-style comment data
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    comments_df = pd.DataFrame(data['comments'])

app = Flask(__name__)
CORS(app)

# Search function: looks for any matching text in comments
def search_comments(query):
    if not query:
        return []

    # Case-insensitive substring match
    results = comments_df[comments_df['text'].str.contains(query, case=False, na=False)]

    # Optional: sort by depth (e.g. prioritize top-level comments)
    results = results.sort_values(by='depth')

    # Return relevant fields
    return results[['author', 'text', 'depth']].head(20).to_dict(orient='records')

# Homepage route
@app.route("/")
def home():
    return render_template('base.html', title="Reddit Movie Comment Search")

# Search endpoint
@app.route("/search")
def search_route():
    query = request.args.get("q", "")
    results = search_comments(query)
    return jsonify(results)

# Run the app locally
if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
'''