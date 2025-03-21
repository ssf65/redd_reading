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
