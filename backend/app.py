import os
import json
import math
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from better_profanity import profanity
import pandas as pd

# Load profanity filter
profanity.load_censor_words()

app = Flask(__name__)
CORS(app)

# Paths
current_directory = os.path.dirname(os.path.abspath(__file__))
grouped_path = os.path.join(current_directory, "init.json")

# Load only the titles and post ID mapping
with open(grouped_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

title_to_id = {}
id_to_meta = {}

for item in raw_data:
    post_id = item["post_url"].split("/")[-1]
    title_to_id[item["title"]] = post_id
    id_to_meta[post_id] = item  # includes 'title', 'comments', etc.

# Load BERT model once
bert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

@app.route("/")
def home():
    return render_template("base.html", title="Reddit Movie Comment Search")

@app.route("/titles")
def get_titles():
    return jsonify([
        {"title": title, "post_id": pid, "num_comments": id_to_meta[pid]["num_comments"]}
        for title, pid in title_to_id.items()
    ])

@app.route("/search_comments")
def search_comments():
    post_id = request.args.get("post_id", "")
    query = request.args.get("q", "")
    filter_flag = request.args.get("filter_offensive", "false").lower() == "true"

    if not post_id or post_id not in id_to_meta or not query.strip():
        return jsonify([])

    comments = id_to_meta[post_id]["comments"]
    df = pd.DataFrame(comments)
    df['text'] = df['text'].fillna("")
    texts = df['text'].tolist()

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = tfidf_scores.argsort()[-100:][::-1].copy()

    # BERT similarity
    candidate_embeddings = bert_model.encode([texts[i] for i in top_indices], convert_to_tensor=True)
    query_embedding = bert_model.encode(query, convert_to_tensor=True)
    bert_scores = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]

    results = []
    for score, i in zip(bert_scores, range(len(top_indices))):
        real_idx = top_indices[i]
        row = df.iloc[real_idx]
        text = row['text']
        if filter_flag:
            text = profanity.censor(text)
        length_boost = math.log(1 + len(text.split()))
        final_score = float(score) * length_boost
        results.append({
            "author": row.get("user", "unknown"),
            "text": text,
            "depth": 0,
            "bert_score": round(float(score), 4),
            "length_boost": round(length_boost, 4),
            "final_score": round(final_score, 4),
            "permalink": row.get("full_permalink", ""),
        })

    return jsonify(sorted(results, key=lambda x: x['final_score'], reverse=True)[:20])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
