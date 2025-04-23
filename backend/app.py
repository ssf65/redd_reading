import os
import json
import math
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from better_profanity import profanity

# Load profanity filter
profanity.load_censor_words()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Load JSON data
current_directory = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_directory, "init.json")
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

title_to_id = {}
id_to_meta = {}

# Build title-ID mapping
for item in raw_data:
    post_id = item["post_url"].split("/")[-1]
    title_to_id[item["title"]] = post_id
    id_to_meta[post_id] = item

@app.route("/")
def home():
    return render_template("base.html", title="Reddit Comment Search")

@app.route("/titles")
def get_titles():
    return jsonify([
        {"title": title, "post_id": pid, "num_comments": id_to_meta[pid]["num_comments"]}
        for title, pid in title_to_id.items()
    ])

@app.route("/search_comments")
def search_comments():
    post_ids = request.args.getlist("post_ids")
    query = request.args.get("q", "")
    filter_flag = request.args.get("filter_offensive", "false").lower() == "true"

    if not post_ids or not query.strip():
        return jsonify([])

    # Collect comments from all selected posts
    all_comments = []
    for pid in post_ids:
        if pid in id_to_meta:
            comments = id_to_meta[pid].get("comments", [])
            for comment in comments:
                if "text" in comment and comment["text"].strip():
                    all_comments.append(comment)

    if not all_comments:
        return jsonify([])

    df = pd.DataFrame(all_comments)
    df["text"] = df["text"].fillna("")

    # Remove deleted or empty comments
    df = df[df["text"].str.strip().str.lower().ne("[deleted]")]
    df = df[df["text"].str.strip() != ""]

    texts = df["text"].tolist()
    if not texts:
        return jsonify([])

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        token_pattern=r"(?u)\b\w\w+\b"
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return jsonify([])

    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = tfidf_scores.argsort()[-100:][::-1].copy()

    # SVD for interpretability
    svd = TruncatedSVD(n_components=5, random_state=42)
    try:
        svd_matrix = svd.fit_transform(tfidf_matrix[top_indices])
    except ValueError:
        return jsonify([])

    components = svd.components_
    feature_names = vectorizer.get_feature_names_out()

    # Generate interpretable labels
    dimension_labels = []
    for comp in components:
        sorted_indices = np.argsort(comp)[::-1]
        filtered_terms = []
        for idx in sorted_indices:
            term = feature_names[idx].lower()
            if term not in ["deleted", "ve", "im", "dont", "didn", "http", "https"]:
                filtered_terms.append(term)
            if len(filtered_terms) == 3:
                break
        dimension_labels.append("/".join(filtered_terms))

    results = []
    for i, idx in enumerate(top_indices[:20]):
        row = df.iloc[idx]
        text = row["text"]
        if filter_flag:
            text = profanity.censor(text)

        doc_vector = svd_matrix[i]
        label_details = {
            f"Dim{d+1}: {dimension_labels[d]}": round(doc_vector[d], 3)
            for d in range(len(dimension_labels))
        }

        length_boost = math.log(1 + len(text.split()))
        final_score = tfidf_scores[idx] * length_boost

        results.append({
            "author": row.get("user", "unknown"),
            "text": text,
            "depth": 0,
            "tfidf_score": round(tfidf_scores[idx], 4),
            "length_boost": round(length_boost, 4),
            "final_score": round(final_score, 4),
            "label_details": label_details,
            "permalink": row.get("full_permalink", "")
        })

    return jsonify(sorted(results, key=lambda r: r["final_score"], reverse=True))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
