import os
import json
import math
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup paths
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load Reddit-style comments
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    comments = data.get("comments", [])
    comments_df = pd.DataFrame(comments)
    comments_df['text'] = comments_df['text'].fillna("")

# TF-IDF vectorization (for initial filtering)
print("Fitting TF-IDF vectorizer...")
texts = comments_df['text'].tolist()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Load BERT model and precompute embeddings
print("Loading MiniLM model...")
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

print("Encoding all comments with BERT...")
comment_embeddings = bert_model.encode(texts, convert_to_tensor=True)

# Flask setup
app = Flask(__name__)
CORS(app)

# Hybrid search with log-scaled length boost
def hybrid_search(query, tfidf_top_k=100, final_top_k=20):
    if not query.strip():
        return []

    # Step 1: TF-IDF filter (get top 100)
    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_tfidf_indices = tfidf_scores.argsort()[-tfidf_top_k:][::-1].copy()

    # Step 2: Use BERT to re-rank those top 100
    candidate_texts = [texts[i] for i in top_tfidf_indices]
    candidate_embeddings = comment_embeddings[top_tfidf_indices]

    query_embedding = bert_model.encode(query, convert_to_tensor=True)
    bert_scores = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]

    # Build results with length-aware score boost
    results = []
    for score, idx in zip(bert_scores, range(len(top_tfidf_indices))):
        real_idx = top_tfidf_indices[int(idx)]
        row = comments_df.iloc[real_idx]

        length = len(row['text'].split())
        length_boost = math.log(1 + length)  # log-scaled length boost
        final_score = float(score) * length_boost

        results.append({
            "author": row.get("author", "unknown"),
            "text": row.get("text", ""),
            "depth": int(row.get("depth", 0)),
            "bert_score": round(float(score), 4),
            "length_boost": round(length_boost, 4),
            "final_score": round(final_score, 4)
        })

    # Sort by boosted score and return top N
    results = sorted(results, key=lambda r: r['final_score'], reverse=True)
    return results[:final_top_k]

# Routes
@app.route("/")
def home():
    return render_template('base.html', title="Reddit Movie Comment Search")

@app.route("/search")
def search_route():
    query = request.args.get("q", "")
    return jsonify(hybrid_search(query))

# Start app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
